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
        print("Failed to load data: {e}")
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

        