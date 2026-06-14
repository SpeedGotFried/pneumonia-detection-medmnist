import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src to system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from dataset import PneumoniaMNIST, get_dataset_info
    print("SUCCESS: Imported dataset utilities.")
except ImportError:
    print("ERROR: Could not import dataset module. Ensure you are running from the project root.")
    sys.exit(1)

def main():
    print("Downloading and loading PneumoniaMNIST dataset...")
    # Load training set without any normalization/resize transforms to see raw 28x28 images
    train_dataset = PneumoniaMNIST(split='train', download=True)
    
    # Get labels and images
    images = train_dataset.imgs     # Shape: [4708, 28, 28]
    labels = train_dataset.labels.flatten() # Shape: [4708]
    
    info = get_dataset_info()
    label_dict = info['label'] # {'0': 'normal', '1': 'pneumonia'}
    
    # Separate indices for each class
    normal_idx = np.where(labels == 0)[0]
    pneumonia_idx = np.where(labels == 1)[0]
    
    # Take 4 samples of each class
    num_samples = 4
    selected_normal = normal_idx[:num_samples]
    selected_pneumonia = pneumonia_idx[:num_samples]
    
    fig, axes = plt.subplots(2, num_samples, figsize=(12, 6))
    
    # Plot normal images
    for i, idx in enumerate(selected_normal):
        axes[0, i].imshow(images[idx], cmap='gray')
        axes[0, i].set_title(f"Normal (Idx: {idx})")
        axes[0, i].axis('off')
        
    # Plot pneumonia images
    for i, idx in enumerate(selected_pneumonia):
        axes[1, i].imshow(images[idx], cmap='gray')
        axes[1, i].set_title(f"Pneumonia (Idx: {idx})")
        axes[1, i].axis('off')
        
    plt.suptitle("PneumoniaMNIST Dataset Samples (Normal vs Pneumonia)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    output_path = "pneumonia_samples.png"
    plt.savefig(output_path, dpi=300)
    print(f"SUCCESS: Saved sample visualization to '{output_path}'.")
    
if __name__ == "__main__":
    main()
