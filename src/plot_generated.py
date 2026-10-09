import numpy as np
import matplotlib.pyplot as plt
import os
import random
import glob

def annotate_peak(ax, x, y, label_prefix="", color="red"):
    xmax = np.argmax(y)
    ymax = y[xmax]
    x_peak = x[xmax]
    
    ax.scatter(x_peak, ymax, color=color, s=50, zorder=5)
    ax.annotate(f'Max: ({x_peak:.0f}, {ymax:.2f})',
                xy=(x_peak, ymax), 
                xytext=(10, 10),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color=color),
                color=color, fontsize=9)

def read_file(path):
    if path.endswith('.dat'):
        with open(path, 'rb') as f:
            content = f.read()
            if len(content) < 4096: return None
            data_bytes = content[-4096:]
            return np.frombuffer(data_bytes, dtype='<u4').astype(np.float64)
    elif path.endswith('.npy'):
        return np.load(path)
    return None

def plot_side_by_side(seed_dir="dataset/seeds", gen_dir="dataset/generated", target_classes=["Background", "Cs137", "I131"]):
    if not os.path.exists(seed_dir) or not os.path.exists(gen_dir):
        print("Error: Dataset directories not found.")
        return

    # Filter available classes based on target_classes
    available_classes = [d for d in os.listdir(gen_dir) if os.path.isdir(os.path.join(gen_dir, d))]
    classes = [c for c in target_classes if c in available_classes]
    
    if not classes:
        print(f"No matching classes found. Available: {available_classes}")
        return

    # Layout: Rows = Classes, Cols = 2 (Left: Seed, Right: Generated)
    fig, axes = plt.subplots(len(classes), 2, figsize=(16, 5 * len(classes)))
    
    # If only one class, axes is 1D array, make it 2D
    if len(classes) == 1:
        axes = [axes]

    for idx, cls in enumerate(classes):
        # Row axes
        ax_seed = axes[idx][0]
        ax_gen = axes[idx][1]
        
        # --- Left Plot: Original Seed ---
        cls_seed_dir = os.path.join(seed_dir, cls)
        seed_files = glob.glob(os.path.join(cls_seed_dir, "*.dat"))
        
        if seed_files:
            seed_file = random.choice(seed_files)
            seed_counts = read_file(seed_file)
            if seed_counts is not None:
                # Apply log(x+1)
                seed_counts = np.log1p(seed_counts)
                
                channels = np.arange(len(seed_counts))
                ax_seed.plot(channels, seed_counts, color='black', alpha=0.9)
                annotate_peak(ax_seed, channels, seed_counts, color='red')
                ax_seed.set_title(f"Class: {cls} - Original Seed (Log Scale)")
        else:
            ax_seed.text(0.5, 0.5, "No Seed File", ha='center')
        
        # --- Right Plot: Generated Sample ---
        cls_gen_dir = os.path.join(gen_dir, cls)
        gen_files = glob.glob(os.path.join(cls_gen_dir, "*.npy"))
        
        if gen_files:
            gen_file = random.choice(gen_files)
            gen_counts = read_file(gen_file)
            if gen_counts is not None:
                # Apply log(x+1)
                gen_counts = np.log1p(gen_counts)
                
                channels = np.arange(len(gen_counts))
                ax_gen.plot(channels, gen_counts, color='blue', alpha=0.9)
                annotate_peak(ax_gen, channels, gen_counts, color='orange')
                ax_gen.set_title(f"Class: {cls} - Generated Sample (Log Scale)")
        else:
            ax_gen.text(0.5, 0.5, "No Generated File", ha='center')

        # Formatting
        for ax in [ax_seed, ax_gen]:
            ax.set_xlabel("Channel")
            ax.set_ylabel("Log(Counts + 1)")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('co57_k40_visualization.png')
    print("Plot saved as 'co57_k40_visualization.png'")

if __name__ == "__main__":
    plot_side_by_side(target_classes=["Co57", "K40"])
