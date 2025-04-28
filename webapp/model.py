import torch 
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import imutils
from torchvision import transforms
import io
import base64
from PIL import Image

val_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

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


def preprocess_image(img_array, img_size=224):
    """
    Preprocess a single image (numpy array) for inference.
    """
    img = crop_img(img_array)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.bilateralFilter(img, 2, 50, 50)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.uint8(img)
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def np_to_base64(img_np):
    img_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    buf.seek(0)
    img_bytes = buf.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_b64

def inference_pipeline(image_file, model):
    """
    This function will return tumor class for the given image using the trained model.
    
    Args:
        image_file (file-like): File-like object containing the image (e.g., from Flask)
        model (nn.Module): Trained brain tumor classification model
        
    Returns:
        str: Predicted tumor class name
    """
    class_list = ['glioma', 'meningioma', 'notumor', 'pituitary']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Read image file into numpy array
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_array is None:
        raise ValueError("Could not decode image")
    
    # Preprocess the image
    img = preprocess_image(img_array)
    img_tensor = val_test_transform(img).unsqueeze(0).to(device)
    
    # Target layer: last convolutional layer
    target_layer = model.base_model.layer4[-1]

    # Hook for activations and gradients
    feature_maps = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal feature_maps
        feature_maps = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # Forward
    outputs = model(img_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    pred_class_idx = probabilities.argmax(dim=1).item()
    pred_class_name = class_list[pred_class_idx]
    pred_prob = probabilities[0, pred_class_idx].item()

    # Backward
    model.zero_grad()
    outputs[0, pred_class_idx].backward()

    # Grad-CAM calculation
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    feature_maps_weighted = feature_maps[0] * pooled_gradients[:, None, None]
    heatmap = feature_maps_weighted.sum(dim=0)
    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.cpu().detach().numpy()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    original_img = crop_img(img_array)
    original_img = cv2.resize(original_img, (224,224))

    # Create heatmap
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    # Convert images to base64
    original_img_b64 = np_to_base64(original_img)
    gradcam_b64 = np_to_base64(superimposed_img)

    return pred_class_name, pred_prob, original_img_b64, gradcam_b64


class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BrainTumorClassifier, self).__init__()
        # Load pre-trained ResNet-50 model with ImageNet weights
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        for param in self.base_model.parameters():
            param.requires_grad = True

        # Replace the final fully connected layer
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)