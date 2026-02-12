import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy.ndimage import median_filter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_folder", type=str, help="Path to result folder containing rc_records.npy")
    parser.add_argument("--epoch", type=int, default=None, help="Specific epoch to inspect (optional)")
    args = parser.parse_args()
    
    result_folder = Path(args.result_folder)
    rc_records_path = result_folder / 'rc_records.npy'
    
    if not rc_records_path.exists():
        print(f"Error: rc_records.npy not found at {rc_records_path}")
        exit(1)
    
    # Load rc_records
    rc_records = np.load(rc_records_path, allow_pickle=True).item()
    
    print("=" * 80)
    print("RC RECORDS INSPECTION")
    print("=" * 80)
    
    print(f"\nAvailable epochs: {sorted(rc_records.keys())}")
    
    # If specific epoch requested
    if args.epoch is not None:
        if args.epoch in rc_records:
            rc_values = rc_records[args.epoch]
            print(f"\nEpoch {args.epoch} RC values:")
            print(f"  Shape: {rc_values.shape}")
            print(f"  Min: {rc_values.min():.6f}")
            print(f"  Max: {rc_values.max():.6f}")
            print(f"  Mean: {rc_values.mean():.6f}")
            print(f"  Std: {rc_values.std():.6f}")
        else:
            print(f"Epoch {args.epoch} not found in rc_records")
            exit(1)
    else:
        # Print statistics for all epochs
        print("\nStatistics for all epochs:")
        print(f"{'Epoch':<10} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
        print("-" * 58)
        
        for epoch in sorted(rc_records.keys()):
            rc_values = rc_records[epoch]
            print(f"{epoch:<10} {rc_values.min():<12.6f} {rc_values.max():<12.6f} {rc_values.mean():<12.6f} {rc_values.std():<12.6f}")
    
    # Plot RC evolution across epochs
    print("\n" + "=" * 80)
    print("PLOTTING RC RECORDS...")
    print("=" * 80)
    
    epochs = sorted(rc_records.keys())
    mean_rcs = [rc_records[ep].mean() for ep in epochs]
    std_rcs = [rc_records[ep].std() for ep in epochs]
    min_rcs = [rc_records[ep].min() for ep in epochs]
    max_rcs = [rc_records[ep].max() for ep in epochs]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mean RC over epochs
    axes[0, 0].plot(epochs, mean_rcs, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].fill_between(epochs, 
                             np.array(mean_rcs) - np.array(std_rcs),
                             np.array(mean_rcs) + np.array(std_rcs),
                             alpha=0.3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('RC Value')
    axes[0, 0].set_title('Mean RC over Epochs (with ±1 Std)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Min/Max RC over epochs
    axes[0, 1].plot(epochs, min_rcs, 'r-o', label='Min', linewidth=2, markersize=6)
    axes[0, 1].plot(epochs, max_rcs, 'g-o', label='Max', linewidth=2, markersize=6)
    axes[0, 1].fill_between(epochs, min_rcs, max_rcs, alpha=0.2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('RC Value')
    axes[0, 1].set_title('Min/Max RC over Epochs')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Histogram of RC values at last epoch
    last_epoch = epochs[-1]
    last_rc = rc_records[last_epoch]
    axes[1, 0].hist(last_rc, bins=50, color='skyblue', edgecolor='black')
    axes[1, 0].set_xlabel('RC Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Distribution of RC at Epoch {last_epoch}')
    axes[1, 0].axvline(last_rc.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean={last_rc.mean():.4f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: RC values for all samples at last epoch
    sample_indices = np.arange(len(last_rc))
    axes[1, 1].scatter(sample_indices, last_rc, alpha=0.5, s=10)
    axes[1, 1].axhline(last_rc.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('RC Value')
    axes[1, 1].set_title(f'RC Values for Each Sample at Epoch {last_epoch}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(result_folder / 'rc_records_analysis.png', dpi=200, bbox_inches='tight')
    print(f"RC analysis plot saved to {result_folder / 'rc_records_analysis.png'}")
    
    # Additional scatter plot: samples with low/high RC
    fig2, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.RdYlGn(last_rc)  # Color by RC value
    scatter = ax.scatter(sample_indices, last_rc, c=last_rc, cmap='RdYlGn', s=20, alpha=0.6, vmin=last_rc.min(), vmax=last_rc.max())
    
    ax.axhline(last_rc.mean(), color='b', linestyle='--', linewidth=2, label=f'Mean={last_rc.mean():.4f}')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('RC Value')
    ax.set_title(f'RC Values with Color Coding (Epoch {last_epoch})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('RC Value')
    
    fig2.tight_layout()
    fig2.savefig(result_folder / 'rc_records_scatter.png', dpi=200, bbox_inches='tight')
    print(f"RC scatter plot saved to {result_folder / 'rc_records_scatter.png'}")
    
    # Median filter plot: Apply median filter with window size 50 to RC values (all samples)
    fig3, ax = plt.subplots(figsize=(12, 6))
    
    # Apply median filter to last epoch RC values
    last_rc_filtered = median_filter(last_rc, size=50)
    
    # Plot all samples
    sample_indices_all = np.arange(len(last_rc))
    ax.plot(sample_indices_all, last_rc, 'b-o', linewidth=2, markersize=6, alpha=0.5, label='Original RC')
    ax.plot(sample_indices_all, last_rc_filtered, 'r-', linewidth=2.5, label='Median Filtered (window=50)')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('RC Value')
    ax.set_title(f'RC Values with Median Filter (window size=50) - All Samples (Epoch {last_epoch})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig3.tight_layout()
    fig3.savefig(result_folder / 'rc_records_median_filtered.png', dpi=200, bbox_inches='tight')
    print(f"RC median filtered plot saved to {result_folder / 'rc_records_median_filtered.png'}")
    
    print("\nDone!")
