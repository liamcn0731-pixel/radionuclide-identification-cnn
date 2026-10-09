import torch
import torch.nn as nn
import torch.optim as optim
from model import SpectralCNN
from dataset_loader import get_dataloaders
import os
import time

def train_model():
    # --- Configuration ---
    GEN_DIR = "dataset/generated"
    SEED_DIR = "dataset/seeds"
    MODEL_SAVE_PATH = "best_model.pth"
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    PATIENCE = 10  # Early Stopping Patience
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Data Setup ---
    print("Initializing Data Loaders...")
    train_loader, val_loader, classes = get_dataloaders(GEN_DIR, SEED_DIR, BATCH_SIZE)
    num_classes = len(classes)
    print(f"Detected Classes: {classes}")
    
    # --- Model Setup ---
    model = SpectralCNN(num_classes=num_classes).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # Added weight_decay to help combat severe overfitting (L2 Regularization)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    # --- Training Loop ---
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    print("Starting Training...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # 1. Training Phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        
        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_val_acc*100:.2f}%")
        
        # 3. Early Stopping & Checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Model saved! (New Best Val Loss: {best_val_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"  -> No improvement. Patience: {early_stop_counter}/{PATIENCE}")
            
            # LR Reduction strategy manual check (if loss plateauing)
            if early_stop_counter == 5:
                print("  -> Reducing Learning Rate to 1e-4")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-4
            
            if early_stop_counter >= PATIENCE:
                print("Early Stopping Triggered.")
                break
                
    total_time = time.time() - start_time
    print(f"Training Complete in {total_time:.1f}s")
    print(f"Best Model saved to: {os.path.abspath(MODEL_SAVE_PATH)}")

    # --- Plotting History ---
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='green')
    plt.title('Validation Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")
    # plt.show() # Optional, comment out if running on server

if __name__ == "__main__":
    train_model()
