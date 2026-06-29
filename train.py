import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# Point Python to your src folder for custom imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from dataset import get_data_loaders
from models import get_model

def train_model(model_name, train_loader, val_loader, device, epochs=5, lr=0.001):
    # Initialize Model, Loss Function, and Optimizer
    print(f"\n==================================================")
    print(f"Starting training for {model_name.upper()}...")
    print(f"==================================================")
    
    use_pretrained = (model_name != "vanilla")
    model = get_model(model_name, pretrained=use_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
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
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] -> Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

    # Save the trained model parameters
    os.makedirs("trained", exist_ok=True)
    save_path = f"trained/{model_name}_best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel {model_name.upper()} successfully saved to {save_path}!")

def main():
    # Setup Device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will run on: {device}")

    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 15
    LEARNING_RATE = 0.001
    MODELS_TO_TRAIN = ["vanilla", "resnet", "inception", "mobilenet"]

    # Load Real Data using MedMNIST helper
    print("Loading data loaders...")
    train_loader, val_loader, _ = get_data_loaders(batch_size=BATCH_SIZE, image_size=128, augment_train=True)

    # Train all models
    for model_name in MODELS_TO_TRAIN:
        train_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=EPOCHS,
            lr=LEARNING_RATE
        )

if __name__ == "__main__":
    main()
