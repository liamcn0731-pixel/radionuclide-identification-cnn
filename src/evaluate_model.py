import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SpectralCNN
from dataset_loader import SpectralDataset
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def evaluate():
    # --- Configuration ---
    SEED_DIR = "dataset/seeds"
    MODEL_PATH = "best_model.pth"
    BATCH_SIZE = 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Load Data ---
    print(f"Loading Test Data from {SEED_DIR}...")
    
    # Force the 3 classes that the model was trained on
    TRAIN_CLASSES = ['Background', 'Cs137', 'I131']
    test_dataset = SpectralDataset(SEED_DIR, classes=TRAIN_CLASSES)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    classes = test_dataset.classes
    print(f"Classes: {classes}")
    
    # --- Load Model ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found. Please train first.")
        return

    model = SpectralCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully.")
    
    # --- Evaluation ---
    all_preds = []
    all_labels = []
    
    print("Running Inference...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # --- Metrics ---
    print("\n" + "="*30)
    print("Classification Report")
    print("="*30)
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix saved as 'confusion_matrix.png'")
    plt.show()

if __name__ == "__main__":
    evaluate()
