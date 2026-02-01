import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from datetime import datetime

import json
import argparse
from pathlib import Path
from datetime import datetime

from utils import load_data

from typing import Union

from tqdm import tqdm

def normalize_image(img: torch.Tensor):
    x_min = img.min()
    x_max = img.max()
    img_norm = (img - x_min) / (x_max - x_min + 1e-8) 
    return img_norm.to(torch.float32)

# 评估数据集和数据加载器（不打乱顺序，并返回索引）
class IndexedTensorDataset(TensorDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return (*data, idx)

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

    
# CAE model
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        
        self.encoder = ImageEncoder()
        self.decoder = ImageDecoder()

    def forward(self, x: torch.Tensor):
         
        latent = self.encoder(x)
        x_recon = self.decoder(latent)

        return x_recon, latent

LEARNING_RATE = 5e-4
BATCH_SIZE = 256  
CLEAN_NUM = 16000
ANOMALY_NUM = 1600

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
    # 记录MSE、RC、平滑RC和AUC
    model.eval()
    rc_list = []
    mse_list = []
    indices_list = []
    with torch.no_grad():
        for data in dataloader:
            img, indices = data
            indices_list.append(indices)
            
            # 前向传播
            output, _ = model(img)
            loss = criterion(output, img)  
            
            # 计算MSE
            mse_loss  = loss.mean(dim=[1,2,3])
            mse_list.append(mse_loss)
            
            # 计算RC
            # 展平
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
        device = DEVICE,
        batch_size = BATCH_SIZE
    ):
    images = images.to(device)
    model = model.to(device)

    eval_dataset = IndexedTensorDataset(images.to(device, torch.float32))
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    rc, mse, indices = eval_epoch(model, eval_loader)

    order = torch.argsort(indices)

    rc  = rc[order]
    mse = mse[order]

    return rc, mse

def train (
        model: nn.Module,
        images: torch.Tensor,
        device = DEVICE,
        batch_size = BATCH_SIZE,
        lam = LAMBDA,
        nepochs = NUM_EPOCHS,
        lr = LEARNING_RATE
    ):
    images = images.to(device)
    model = model.to(device)

    train_dataset = TensorDataset(images)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    all_indices = np.arange(len(images))
    reference_indices = np.random.choice(all_indices, size=len(images) // 50, replace=False)
    reference_images = images[reference_indices].to(device)

    model.eval()
    with torch.no_grad():
        _, reference_latent = model(reference_images)
    reference_latent = reference_latent.detach()  # shape: (500, 256)

    pbar = tqdm(range(1, nepochs + 1), desc=f"Training CAE lambda={lam}")
    for epoch in pbar:
        model.train()
        total_loss = 0
        for data, in train_loader:
            img = data
            
            output, latent = model(img)  
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
    try:
        for k,v in config['training_data'].items():
            image_list.append(data[k]['image'][slice(*v)])

    except Exception as e:
        print(f"Failed to construct training data: {e}")
        sys.exit(1)

    result_folder.mkdir(parents=True, exist_ok=True)
    images_tensor = torch.from_numpy(np.concatenate(image_list))
    images_tensor = normalize_image(images_tensor.to(DEVICE)).permute(0,3,1,2)

    print(f'Traning data shape {list(images_tensor.shape)}')

    record_interval = config.get('record_interval', RECORD_INTERVAL)

    train_config = config.get('train', {})
    eval_config = config.get('eval', {})
                
    # 创建结果目录
    result_path = result_folder
    result_path.mkdir(parents=True, exist_ok=True)

    model = CAE()

    loss_records = []
    rc_records = {}
    mse_loss_records = {}

    for epoch, loss in enumerate(train(model, images_tensor, **train_config), 1):

        loss_records.append(loss)

        if epoch % record_interval == 0:
            rc_values, mse_loss_values = eval(model, images_tensor, **eval_config)
            
            rc_values = rc_values.detach().cpu().numpy()
            mse_loss_values = mse_loss_values.detach().cpu().numpy()
            
            mse_loss_records[epoch] = mse_loss_values
            rc_records[epoch] = rc_values

            tqdm.write(f"Saved record in epoch {epoch}")

    # 保存所有记录到文件
    np.save(result_path / 'loss_records.npy', np.array(loss_records))
    np.save(result_path / 'mse_loss_records.npy', mse_loss_records)
    np.save(result_path / 'rc_records.npy', rc_records)
    print(f"Results are saved to {result_path}")

        