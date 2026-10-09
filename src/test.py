import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import SpectralCNN
from dataset_loader import SpectralDataset
import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

def test_new_data():
    # --- Configuration ---
    # Path to the new data folder
    TEST_DIR = "dataset/New"
    MODEL_PATH = "best_model.pth"
    BATCH_SIZE = 64
    
    # Classes used during training (Must match training exactly)
    TRAIN_CLASSES = ['Background', 'Cs137', 'I131']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Validations ---
    if not os.path.exists(TEST_DIR):
        print(f"Error: Directory {TEST_DIR} does not exist.")
        return
        
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model {MODEL_PATH} not found.")
        return

    # --- Load Data ---
    print(f"Loading Data from {TEST_DIR}...")
    # Force class mapping to match training
    test_dataset = SpectralDataset(TEST_DIR, classes=TRAIN_CLASSES)
    
    if len(test_dataset) == 0:
        print("No files found in dataset/New. Please ensure data is organized in subfolders (e.g., dataset/New/Cs137).")
        return
        
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Load Model ---
    model = SpectralCNN(num_classes=len(TRAIN_CLASSES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    # --- Inference ---
    all_preds = []
    all_labels = []
    all_probs = []
    file_paths = []
    
    print("Running Inference...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Get probabilities
            probs = F.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(max_probs.cpu().numpy())
            
            # Get file names for this batch
            # Dataset indices: i*BATCH to (i+1)*BATCH
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + len(labels)
            batch_files = [os.path.basename(f) for f in test_dataset.file_list[start_idx:end_idx]]
            file_paths.extend(batch_files)

    # --- Results ---
    acc = accuracy_score(all_labels, all_preds)
    print("\n" + "="*30)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print("="*30)
    
    # Detailed Report
    target_labels = list(range(len(TRAIN_CLASSES)))
    print(classification_report(all_labels, all_preds, labels=target_labels, target_names=TRAIN_CLASSES, zero_division=0))
    
    # --- Save Individual Predictions ---
    results_df = pd.DataFrame({
        'Filename': file_paths,
        'True_Class': [TRAIN_CLASSES[l] for l in all_labels],
        'Predicted_Class': [TRAIN_CLASSES[p] for p in all_preds],
        'Confidence': all_probs,
        'Correct': [l == p for l, p in zip(all_labels, all_preds)]
    })
    
    save_path = "test_results.csv"
    results_df.to_csv(save_path, index=False)
    print(f"Detailed results saved to '{save_path}'")
    
    # Show some examples of errors if any
    errors = results_df[~results_df['Correct']]
    if not errors.empty:
        print("\nExamples of Incorrect Predictions:")
        print(errors.head())
    else:
        print("\nAll predictions are correct!")

if __name__ == "__main__":
    test_new_data()
