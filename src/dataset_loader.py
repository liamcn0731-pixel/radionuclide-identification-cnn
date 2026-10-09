import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob

def preprocess_spectrum(counts):
    """
    Standardization Pipeline:
    1. Log Transformation: y = log(x + 1)
    2. Min-Max Normalization: y_norm = (y - min) / (max - min)
    """
    # 1. Log Transform
    # Improves weak peaks visibility
    y_log = np.log1p(counts)
    
    # 2. Min-Max Normalization -> [0, 1]
    min_val = np.min(y_log)
    max_val = np.max(y_log)
    
    if max_val - min_val == 0:
        y_norm = np.zeros_like(y_log)
    else:
        y_norm = (y_log - min_val) / (max_val - min_val)
        
    return y_norm.astype(np.float32)

class SpectralDataset(Dataset):
    """
    Dataset class for loading Gamma Spectra.
    Can load from 'dataset/generated' (Training) or 'dataset/seeds' (Validation/Test).
    """
    def __init__(self, root_dir, classes=None):
        self.root_dir = root_dir
        self.file_list = []
        self.labels = []
        self.classes = classes
        
        # If classes not provided, scan directory
        if not self.classes:
            self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            # Filter out empty or hidden folders if necessary
            self.classes = [c for c in self.classes if glob.glob(os.path.join(root_dir, c, "*"))]

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"Loading data from {root_dir}")
        print(f"Classes: {self.class_to_idx}")
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            
            # Find all relevant files (.dat or .npy)
            files = glob.glob(os.path.join(cls_dir, "*.npy")) + glob.glob(os.path.join(cls_dir, "*.dat"))
            
            for f_path in files:
                self.file_list.append(f_path)
                self.labels.append(self.class_to_idx[cls_name])
                
        print(f"Found {len(self.file_list)} samples.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]
        
        # Read file
        if file_path.endswith('.npy'):
            counts = np.load(file_path)
        elif file_path.endswith('.dat'):
            with open(file_path, 'rb') as f:
                content = f.read()
                data_bytes = content[-4096:]
                counts = np.frombuffer(data_bytes, dtype='<u4').astype(np.float64)
        
        # Preprocess
        # Ensure it is 1024 channels
        if len(counts) != 1024:
            # Handle edge cases or resize? For now assume valid data
            counts = counts[:1024] 
            
        processed_data = preprocess_spectrum(counts)
        
        # Reshape to (1, 1024) for CNN
        # PyTorch expects (Channels, Length)
        data_tensor = torch.from_numpy(processed_data).unsqueeze(0) 
        
        return data_tensor, label

def get_dataloaders(gen_dir, seed_dir, batch_size=64):
    """
    Creates DataLoaders for Training (Generated) and Validation (Seeds).
    """
    # Verify both dirs exist
    if not os.path.exists(gen_dir) or not os.path.exists(seed_dir):
        raise FileNotFoundError("Dataset directories (generated or seeds) not found.")
        
    # Create Datasets
    # We want to ensure robust class mapping consistent across both
    # Scan generated dir to get canonical class list
    temp_dataset = SpectralDataset(gen_dir)
    classes = temp_dataset.classes
    
    train_dataset = SpectralDataset(gen_dir, classes=classes)
    val_dataset = SpectralDataset(seed_dir, classes=classes)
    
    # Transformers / DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, classes

if __name__ == "__main__":
    # Test Loader
    train_loader, val_loader, classes = get_dataloaders("dataset/generated", "dataset/seeds")
    data, label = next(iter(train_loader))
    print("Batch Shape:", data.shape)
    print("Label Shape:", label.shape)
    print("Preprocessing check range:", [data.min().item(), data.max().item()])
