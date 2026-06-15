import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO

#  Point Python to your src folder for custom imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from dataset import get_transforms
from models import get_model

def train():
    #  Setup Device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will run on: {device}")

    #  Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 0.001
    MODEL_NAME = "resnet" # You can switch this to "vanilla", "inception", etc.

    # Load Real Data using MedMNIST
    info = INFO['pneumoniamnist']
    DataClass = getattr(medmnist, info['python_class'])
    
    # Get any preprocessing transforms you defined in src/dataset.py
    # (Falling back to default tensor conversion if needed)
    try:
        train_transform, val_transform = get_transforms()
    except:
        from torchvision import transforms
        train_transform = transforms.ToTensor()
        val_transform = transforms.ToTensor()

    train_dataset = DataClass(split='train', transform=train_transform, download=True)
    val_dataset = DataClass(split='val', transform=val_transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #  Initialize Model, Loss Function, and Optimizer
    model = get_model(MODEL_NAME, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #  The Core Training Loop
    print(f"\n Starting training for {MODEL_NAME.upper()}...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze().long() # MedMNIST labels come as 2D arrays, flatten them
            
            # Zero out previous gradients
            optimizer.zero_grad()
            
            # Forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # Perform a single optimization step (parameter update)
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] -> Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

    #  Save the trained model parameters
    os.makedirs("saved_models", exist_ok=True)
    save_path = f"saved_models/{MODEL_NAME}_best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n Model successfully saved to {save_path}!")

if __name__ == "__main__":
    train()
