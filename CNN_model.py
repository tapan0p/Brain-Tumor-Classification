import torch
import torch.nn as nn 
import torch.optim as optim 

class CNN_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(32*56*56,4)
        )

    def forward(self,x):
        return self.model(x)
    
    def train(self,train_loader,valid_loader,lr,epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(),lr=lr)

        for epoch in range(epochs):
            total_train_loss =0
            train_accuracy = 0
            for image,target in train_loader:
                optimizer.zero_grad()
                output =self.forward(image)
                loss =criterion(output,target)
                loss.backward()
                optimizer.step()
                total_train_loss +=loss.item()
                _,predicted = output.max(1)
                train_accuracy += predicted.eq(target).sum().item()

            print(f"Epoch {epoch+1}/{epochs}, training loss = {total_train_loss/len(train_loader)}, training accuracy = {train_accuracy/len(train_loader)}")

            with torch.no_grad():
                total_valid_loss=0 
                valid_accuracy = 0
                for image, target in valid_loader:
                    output = self.forward(image)
                    loss = criterion(output,target)
                    loss.backward()
                    optimizer.step()
                    total_valid_loss += loss.item()
                    _,predicted = output.max(1) 
                    valid_accuracy += predicted.eq(target).sum().item()
                print(f"Epoch {epoch+1}/{epochs},validation loss = {total_valid_loss/len(valid_loader)}, validation accuracy = {valid_accuracy/len(valid_loader)}")
        print("Training complete")


            
