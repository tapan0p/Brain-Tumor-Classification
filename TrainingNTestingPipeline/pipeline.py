from PIL import Image
import cv2
import imutils
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_pil_image
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time



def crop_img(img):
    """
    Finds the extreme points on the image and crops the rectangular region.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return img  
    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = 5
    y1, y2 = max(0, extTop[1] - ADD_PIXELS), min(img.shape[0], extBot[1] + ADD_PIXELS)
    x1, x2 = max(0, extLeft[0] - ADD_PIXELS), min(img.shape[1], extRight[0] + ADD_PIXELS)
    new_img = img[y1:y2, x1:x2].copy()
    return new_img

def preprocess_image(image_path, img_size=224):
    """
    Preprocess a single image for inference.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to load image from {image_path}")
    img = crop_img(img)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.bilateralFilter(img, 2, 50, 50)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.uint8(img)
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def creating_dataset(path_name):
    class_list = sorted(os.listdir(path_name))
    X = []
    y = []
    
    for tumor_type in class_list:
        tumor_path = os.path.join(path_name, tumor_type)
        
        for image in os.listdir(tumor_path):
            image_path = os.path.join(tumor_path, image)
            img = preprocess_image(image_path)
            X.append(img)
            y.append(class_list.index(tumor_type))

    # Convert to PyTorch tensors
    X = torch.FloatTensor(np.array(X)).permute(0, 3, 1, 2) # (N, H, W, C) → (N, C, H, W)
    y = torch.LongTensor(y) 

    return X, y


class AugmentedDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # Get the image tensor
        if self.transform:
            img = self.transform(img)  # Apply augmentation
        return img, self.labels[idx]


class Pipeline:
    def __init__(self):
          super().__init__()
          self.train_transform = transforms.Compose([
            transforms.ToPILImage(),  
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
        ])
          self.val_test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])

    def load_data(self,train_data_path,test_data_path,batch_size):
        X_train, y_train = creating_dataset(train_data_path)
        X_test, y_test = creating_dataset(test_data_path)
        X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=1)
        # Create datasets with transforms
        train_dataset = AugmentedDataset(X_train, y_train, transform=self.train_transform)
        val_dataset = AugmentedDataset(X_val, y_val, transform=self.val_test_transform)
        test_dataset = AugmentedDataset(X_test, y_test, transform=self.val_test_transform)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return train_loader,val_loader,test_loader
    
    def evaluate_model(self,model, loader, criterion, device):
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        return accuracy, running_loss / len(loader)

                
                


    
    def train_model(self, model, train_loader, val_loader, num_epochs, lr):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-5)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        # Lists to store metrics for plotting
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        for epoch in range(num_epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()  
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_acc = 100 * correct / total
            val_acc, val_loss = self.evaluate_model(model, val_loader, criterion, device)

            # Store metrics for plotting
            train_losses.append(running_loss / len(train_loader))
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            scheduler.step()

            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%")
        
        

        return model
    

    def test_model(self,model, test_loader, class_names):
        """Evaluate the model on the test set and plot the confusion matrix."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        y_true, y_pred = [], []
        total_time = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                start_time = time.time()
                outputs = model(images)
                end_time = time.time()
                
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

                total_time += (end_time - start_time)

        # Compute Metrics
        test_acc = accuracy_score(y_true, y_pred) * 100
        precision = precision_score(y_true, y_pred, average="weighted") * 100
        recall = recall_score(y_true, y_pred, average="weighted") * 100
        f1 = f1_score(y_true, y_pred, average="weighted") * 100
        avg_inference_time = total_time / len(test_loader.dataset)

        # Print overall metrics
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Weighted Precision: {precision:.2f}%")
        print(f"Weighted Recall: {recall:.2f}%")
        print(f"Weighted F1 Score: {f1:.2f}%")
        print(f"Avg Inference Time per Image: {avg_inference_time * 1000:.6f} ms")

        # Print class-wise metrics
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    def save_model(self,model,model_name):
        torch.save(model.state_dict(), f"{model_name}.pth")


        
        