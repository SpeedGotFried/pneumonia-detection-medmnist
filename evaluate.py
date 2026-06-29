import os
import sys
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from dataset import get_data_loaders
from models import get_model

def evaluate_model(model_name, test_loader, device):
    checkpoint_path = f"trained/{model_name}_best.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint for {model_name} not found at {checkpoint_path}")
        return None
    
    # Load model
    model = get_model(model_name, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = [] # probability for AUC-ROC
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze().long()
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy()) # probability of class 1 (pneumonia)
            
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)
    
    # Specificity: True Negatives / (True Negatives + False Positives)
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1-Score": f1,
        "AUC-ROC": auc
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating models on: {device}")
    
    # Load data loaders (evaluating on test set)
    print("Loading test dataset...")
    _, _, test_loader = get_data_loaders(batch_size=64, image_size=128, augment_train=False)
    
    models = ["vanilla", "resnet", "inception", "mobilenet"]
    results = {}
    
    for model_name in models:
        print(f"Evaluating {model_name}...")
        metrics = evaluate_model(model_name, test_loader, device)
        if metrics:
            results[model_name] = metrics
            
    # Print results in a neat format
    print("\n" + "="*80)
    print(f"{'Model':15s} | {'Accuracy':9s} | {'Precision':9s} | {'Recall':9s} | {'Specificity':11s} | {'F1-Score':8s} | {'AUC-ROC':8s}")
    print("="*80)
    for model_name, metrics in results.items():
        print(f"{model_name.upper():15s} | "
              f"{metrics['Accuracy']*100:8.2f}% | "
              f"{metrics['Precision']*100:8.2f}% | "
              f"{metrics['Recall']*100:8.2f}% | "
              f"{metrics['Specificity']*100:10.2f}% | "
              f"{metrics['F1-Score']*100:7.2f}% | "
              f"{metrics['AUC-ROC']*100:7.2f}%")
    print("="*80)

if __name__ == "__main__":
    main()
