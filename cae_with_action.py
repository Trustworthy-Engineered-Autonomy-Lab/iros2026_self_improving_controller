import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from datetime import datetime
from utils import IndexedTensorDataset

import json
import argparse
from pathlib import Path
from datetime import datetime

from utils import load_data, normalize_image

from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

class ImageEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Flatten(),
            nn.Linear(64 * 18 * 28, 256),
            nn.ReLU(True)
        )

class ImageDecoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(256, 64 * 18 * 28),
            nn.ReLU(True),
            nn.Unflatten(1, (64, 18, 28)),

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

class SteerEncoder(nn.Sequential):
    """Encode steering into latent space"""
    def __init__(self):
        super().__init__(
            nn.Linear(1, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True)
        )
    
# CAE model
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        
        self.encoder = ImageEncoder()
        self.steer_encoder = SteerEncoder()
        self.decoder = ImageDecoder()

    def forward(self, image: torch.Tensor, steer: torch.Tensor):
        """
        Args:
            image: (B, 3, 144, 224)
            steer: (B, 1)
        Returns:
            x_recon: (B, 3, 144, 224)
            latent: (B, 256)
        """
        image_latent = self.encoder(image)  # (B, 256)
        steer_latent = self.steer_encoder(steer)  # (B, 256)
        
        # Combine latents by averaging
        latent = (image_latent + steer_latent) / 2.0
        x_recon = self.decoder(latent)

        return x_recon, latent

LEARNING_RATE = 5e-4
BATCH_SIZE = 256  

NUM_EPOCHS = 100
RECORD_INTERVAL = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULT_FOLDER = "cae_%Y_%m_%d_%H_%M_%S"
LAMBDA = 1


def eval_epoch(
        model: nn.Module,
        dataloader: DataLoader
    ):
    
    criterion = nn.MSELoss(reduction='none')
    model.eval()
    rc_list = []
    mse_list = []
    indices_list = []
    with torch.no_grad():
        for data in dataloader:
            img, steer, indices = data
            indices_list.append(indices)
            
            # Forward pass
            output, _ = model(img, steer)
            loss = criterion(output, img)  
            
            # Calculate MSE
            mse_loss  = loss.mean(dim=[1,2,3])
            mse_list.append(mse_loss)
            
            # Calculate RC
            img_flat = img.view(img.size(0), -1) 
            output_flat = output.view(output.size(0), -1)
            
            img_mean = img_flat.mean(dim=1, keepdim=True)
            output_mean = output_flat.mean(dim=1, keepdim=True)
            
            img_centered = img_flat - img_mean 
            output_centered = output_flat - output_mean
            
            numerator = torch.sum(img_centered * output_centered, dim=1) 
            
            img_norm = torch.norm(img_centered, p=2, dim=1) 
            output_norm = torch.norm(output_centered, p=2, dim=1)
            
            denominator = img_norm * output_norm + 1e-8
            rc = numerator / denominator
            rc_list.append(rc)

    return torch.cat(rc_list), torch.cat(mse_list), torch.cat(indices_list)
    
def eval (
        model: nn.Module,
        images: torch.Tensor,
        steers: torch.Tensor,
        device = DEVICE,
        batch_size = BATCH_SIZE
    ):
    images = images.to(device, torch.float32)
    steers = steers.to(device, torch.float32)
    model = model.to(device)

    class IndexedDataset(Dataset):
        def __init__(self, images, steers):
            self.images = images
            self.steers = steers
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            return self.images[idx], self.steers[idx], idx

    eval_dataset = IndexedDataset(images, steers)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    rc, mse, indices = eval_epoch(model, eval_loader)

    order = torch.argsort(indices)

    rc  = rc[order]
    mse = mse[order]

    return rc, mse

def train (
        model: nn.Module,
        images: torch.Tensor,
        steers: torch.Tensor,
        device = DEVICE,
        batch_size = BATCH_SIZE,
        lam = LAMBDA,
        nepochs = NUM_EPOCHS,
        lr = LEARNING_RATE
    ):
    images = images.to(device, torch.float32)
    steers = steers.to(device, torch.float32)
    model = model.to(device)

    train_dataset = TensorDataset(images, steers)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    all_indices = np.arange(len(images))
    reference_indices = np.random.choice(all_indices, size=len(images) // 50, replace=False)
    reference_images = images[reference_indices]
    reference_steers = steers[reference_indices]

    model.eval()
    with torch.no_grad():
        _, reference_latent = model(reference_images, reference_steers)
    reference_latent = reference_latent.detach()

    pbar = tqdm(range(1, nepochs + 1), desc=f"Training CAE with Steering lambda={lam}")
    for epoch in pbar:
        model.train()
        total_loss = 0
        for img, steer in train_loader:
            output, latent = model(img, steer)  
            loss_recon = criterion(output, img)  

            mse_loss = loss_recon.mean(dim=[1,2,3])  

            distances = torch.cdist(latent, reference_latent)  

            min_indices = torch.argmin(distances, dim=1)  
            nearest_latent = reference_latent[min_indices]  

            l2_loss = (latent - nearest_latent).pow(2).mean(dim=1)  

            total_loss_batch = mse_loss + lam * l2_loss

            loss_mean = total_loss_batch.mean()

            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()

            total_loss += loss_mean.item()
        
        avg_loss = total_loss / len(train_loader)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

        yield avg_loss


def plot_rc_records(rc_records: dict, result_folder: Path):
    """
    Plot RC records analysis after training completes
    
    Args:
        rc_records: Dictionary of epoch -> rc_values array
        result_folder: Path to save the plots
    """
    print("\n" + "=" * 80)
    print("PLOTTING RC RECORDS ANALYSIS...")
    print("=" * 80)
    
    epochs = sorted(rc_records.keys())
    mean_rcs = [rc_records[ep].mean() for ep in epochs]
    std_rcs = [rc_records[ep].std() for ep in epochs]
    min_rcs = [rc_records[ep].min() for ep in epochs]
    max_rcs = [rc_records[ep].max() for ep in epochs]
    
    # Print statistics for all epochs
    print("\nStatistics for all epochs:")
    print(f"{'Epoch':<10} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
    print("-" * 58)
    
    for epoch in epochs:
        rc_values = rc_records[epoch]
        print(f"{epoch:<10} {rc_values.min():<12.6f} {rc_values.max():<12.6f} {rc_values.mean():<12.6f} {rc_values.std():<12.6f}")
    
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
    
    print("\nRC Records Analysis Complete!")
                        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()  

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print(f"Failed to load config file {args.config} : {e}")
        sys.exit(1)

    record_interval = config.get("record_interval", RECORD_INTERVAL)
    result_folder = Path(datetime.now().strftime(config.get("result_folder", RESULT_FOLDER)))
    
    try:
        data = load_data(config['load_data'])
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

    image_list = []
    steer_list = []
    try:
        for k,v in config['training_data'].items():
            image_list.append(data[k]['image'][slice(*v)])
            steer_list.append(data[k]['steer'][slice(*v)])

    except Exception as e:
        print(f"Failed to construct training data: {e}")
        sys.exit(1)

    images_tensor = torch.from_numpy(np.concatenate(image_list))
    images_tensor = normalize_image(images_tensor.to(DEVICE)).permute(0,3,1,2)
    
    steer_tensor = torch.from_numpy(np.concatenate(steer_list)).float().unsqueeze(1)
    
    # Normalize steering to [-1, 1]
    steer_tensor = torch.clamp(steer_tensor / 1.0, -1, 1)

    print(f'Training data shape: images {list(images_tensor.shape)}, steering {list(steer_tensor.shape)}')

    record_interval = config.get('record_interval', RECORD_INTERVAL)

    train_config = config.get('train', {})
    eval_config = config.get('eval', {})
                
    result_folder.mkdir(parents=True, exist_ok=True)

    model = CAE()

    loss_records = []
    rc_records = {}
    mse_loss_records = {}

    for epoch, loss in enumerate(train(model, images_tensor, steer_tensor, **train_config), 1):

        loss_records.append(loss)

        if epoch % record_interval == 0:
            rc_values, mse_loss_values = eval(model, images_tensor, steer_tensor, **eval_config)
            
            rc_values = rc_values.detach().cpu().numpy()
            mse_loss_values = mse_loss_values.detach().cpu().numpy()
            
            mse_loss_records[epoch] = mse_loss_values
            rc_records[epoch] = rc_values

            tqdm.write(f"Saved record in epoch {epoch}")

    # 保存所有记录到文件
    np.save(result_folder / 'loss_records.npy', loss_records)
    np.save(result_folder / 'mse_loss_records.npy', mse_loss_records)
    np.save(result_folder / 'rc_records.npy', rc_records)
    torch.save(model.encoder.state_dict(), result_folder / 'encoder.pt')
    torch.save(model.decoder.state_dict(), result_folder / 'decoder.pt')
    print(f"Results are saved to {result_folder}")
    
    # Generate RC records analysis plots
    plot_rc_records(rc_records, result_folder)

        