import os
import numpy as np
import glob
import shutil
import random

def read_dat_file(file_path):
    """
    Reads a .dat file and returns the counts (last 1024 channels).
    Based on datatest.py logic.
    """
    with open(file_path, 'rb') as f:
        content = f.read()

    # Extract counts (last 1024 channels, uint32)
    # 1024 * 4 bytes = 4096 bytes
    if len(content) < 4096:
        print(f"Warning: File {file_path} is too small. Skipping.")
        return None
        
    data_bytes = content[-4096:]
    counts = np.frombuffer(data_bytes, dtype='<u4')
    
    # Return as float for processing
    return counts.astype(np.float64)

class SpectraAugmentor:
    """
    Implements on-the-fly augmentation for gamma spectra.
    """
    def __init__(self, target_counts_range=(50000, 150000), gain_scale_range=(0.99, 1.01), shift_range=(-2.0, 2.0)):
        self.target_counts_range = target_counts_range
        self.gain_scale_range = gain_scale_range
        self.shift_range = shift_range

    def poisson_resample(self, counts):
        """
        Normalizes spectrum to PDF and resamples aiming for a target total count.
        """
        total_counts = np.sum(counts)
        if total_counts == 0:
            return counts
            
        # Normalize to PDF
        pdf = counts / total_counts
        
        # Pick a random target total count
        target_total = np.random.randint(self.target_counts_range[0], self.target_counts_range[1] + 1)
        
        # Resample
        # Method: scale counts to target sum, then add Poisson noise?
        # Or: Use np.random.multinomial then add poisson noise?
        # Standard way: Scale expected counts per channel, then draw from Poisson
        
        expected_counts = pdf * target_total
        resampled = np.random.poisson(expected_counts)
        return resampled.astype(np.float64)

    def gain_shift(self, counts):
        """
        Applies gain (scaling) and shift (translation) to the spectrum.
        x' = x * (1 +/- 0.02)
        x' = x +/- 3
        """
        n_channels = len(counts)
        channels = np.arange(n_channels)
        
        # Random params
        scale = np.random.uniform(self.gain_scale_range[0], self.gain_scale_range[1])
        shift = np.random.uniform(self.shift_range[0], self.shift_range[1])
        
        # New channel positions
        new_channels = channels * scale + shift
        
        # Interpolate back to original integer channels [0, 1023]
        # We need to preserve total counts roughly, but interpolation changes shape
        # Better to treat as value interpolation? No, spectrum is histogram.
        # Warping the x-axis.
        
        # Linear interpolation
        # new_y(x) = interp(x, new_channels, old_counts)
        # But we need to define new_channels such that new_channels[i] maps to original counts
        
        # Correct approach for histogram warping:
        # The signal at original channel `i` moves to `i * scale + shift`.
        # We want to know what lands in integer bins 0..1023.
        
        augmented_counts = np.zeros_like(counts)
        
        # Simple interpolation (fast, approximate)
        # np.interp(x_new, x_old, y_old)
        # We want value at grid points 0..1023
        # The ORIGINAL calibrated function was y = f(x)
        # The NEW function is y' = f( (x - shift)/scale )
        
        # Let's use np.interp
        # We want to sample at integer indices `xi`
        # Correspond to source coordinates `src_x = (xi - shift) / scale`
        
        target_x = np.arange(n_channels)
        src_x = (target_x - shift) / scale
        
        # Use simple linear interpolation of counts
        # Note: This doesn't strictly preserve integral, but for small shifts it's okay for CNNs usually.
        # For rigorous physics, we'd accumulate bin overlaps.
        augmented_counts = np.interp(src_x, channels, counts, left=0, right=0)
        
        return augmented_counts

    def augment(self, counts):
        # 1. Poisson Resampling
        counts_aug = self.poisson_resample(counts)
        # 2. Gain Shifting
        counts_aug = self.gain_shift(counts_aug)
        return counts_aug

def main():
    # Configuration
    SEED_DIR = "dataset/seeds"
    OUTPUT_DIR = "dataset/generated"
    SAMPLES_PER_CLASS = 180
    
    augmentor = SpectraAugmentor()
    
    # Check if seed dir exists
    if not os.path.exists(SEED_DIR):
        print(f"Error: {SEED_DIR} does not exist.")
        print("Please create it and organize your .dat files into subdirectories (e.g., A, B, Background).")
        # Create a dummy structure for user convenience
        os.makedirs(os.path.join(SEED_DIR, "Cs137"), exist_ok=True)
        os.makedirs(os.path.join(SEED_DIR, "I131"), exist_ok=True)
        os.makedirs(os.path.join(SEED_DIR, "Background"), exist_ok=True)
        print(f"Created placeholder directories in {SEED_DIR}. Please populate them.")
        return

    # Clean output dir
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # Process each class
    classes = [d for d in os.listdir(SEED_DIR) if os.path.isdir(os.path.join(SEED_DIR, d))]
    
    if not classes:
        print("No class directories found in seed folder.")
        return

    total_generated = 0
    
    for cls in classes:
        print(f"Processing class: {cls}")
        class_seed_dir = os.path.join(SEED_DIR, cls)
        class_out_dir = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(class_out_dir, exist_ok=True)
        
        seed_files = glob.glob(os.path.join(class_seed_dir, "*.dat"))
        if not seed_files:
            print(f"  No .dat files found in {class_seed_dir}. Skipping.")
            continue
            
        print(f"  Found {len(seed_files)} seed files.")
        
        # Generate 180 samples
        for i in range(SAMPLES_PER_CLASS):
            # 1. Randomly insert a seed file
            seed_file = random.choice(seed_files)
            original_counts = read_dat_file(seed_file)
            
            if original_counts is None:
                continue
                
            # 2. Augment
            aug_counts = augmentor.augment(original_counts)
            
            # 3. Save
            # Save as numpy array (.npy) is efficient standard for ML
            save_path = os.path.join(class_out_dir, f"aug_{i:04d}.npy")
            np.save(save_path, aug_counts)
            
        print(f"  Generated {SAMPLES_PER_CLASS} samples in {class_out_dir}")
        total_generated += SAMPLES_PER_CLASS

    print("="*30)
    print(f"Data generation complete! Total samples: {total_generated}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
