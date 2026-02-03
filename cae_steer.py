import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split

import numpy as np
from tqdm import tqdm
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from utils import load_data, construct_data, normalize_image, plot_loss_curve, EarlyStopCriterion
from cae import ImageEncoder, ImageDecoder

class SteerDecoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
            nn.Tanh()
        )

class SteerEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(1, 16),
            nn.Linear(16, 64),
            nn.Linear(64, 128),
            nn.Linear(128, 256),
            nn.Tanh()
        )
    
class CAESteer(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = ImageEncoder()
        self.decoder = SteerDecoder()

    def forward(self, x):
        latent = self.encoder(x)
        steer = self.decoder(latent)

        return steer, latent

LEARNING_RATE = 5e-4
BATCH_SIZE = 256  
WEIGHT_DECAY = 5e-4

NUM_EPOCHS = 100
RECORD_INTERVAL = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULT_FOLDER = "cae_%Y_%m_%d_%H_%M_%S"
LAMBDA = 1
    
def val_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module
    ):
    
    model.eval()
    total_loss = 0
    for image, steer in dataloader:
        with torch.no_grad():
            pred_steer, latent = model(image)
            loss = criterion(pred_steer, steer).mean()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer
    ):
    model.train()
    total_loss = 0
    for image, steer in dataloader:
        
        output, latent = model(image)  
        loss = criterion(output, steer)  

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)

    return avg_loss

def train (
        model: nn.Module,
        images: torch.Tensor,
        steers: torch.Tensor,
        device = DEVICE,
        batch_size = BATCH_SIZE,
        nepochs = NUM_EPOCHS,
        lr = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY
    ):
    images = images.to(device, torch.float32)
    steers = steers.to(device, torch.float32)
    model = model.to(device)

    dataset = TensorDataset(images, steers)

    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    pbar = tqdm(range(1, nepochs + 1), desc=f"Training CAE steer")
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = val_epoch(model, val_loader, criterion)

        pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

        yield train_loss, val_loss

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

    result_folder = Path(datetime.now().strftime(config.get("result_folder", RESULT_FOLDER)))
    
    try:
        data = load_data(config['load_data'])
    except Exception as e:
        print("Failed to load data: {e}")
        sys.exit(1)

    training_data = construct_data(data, config.get('training_data', {}))

    images = torch.from_numpy(training_data['image'])
    steers = torch.from_numpy(training_data['steer']).unsqueeze(1)
    images = normalize_image(images.to(DEVICE)).permute(0,3,1,2)

    print(f'Traning data shape {list(images.shape)}')

    record_interval = config.get('record_interval', RECORD_INTERVAL)

    train_config = config.get('train', {})
    eval_config = config.get('eval', {})
                
    # 创建结果目录
    result_folder.mkdir(parents=True, exist_ok=True)

    model = CAESteer()
    criterion = EarlyStopCriterion()

    train_loss_list = []
    val_loss_list = []
    for epoch, (train_loss, val_loss) in enumerate(train(model, images, steers, **train_config), 1):
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        if criterion(train_loss, val_loss):
            torch.save(model.state_dict(), result_folder / 'best.pt')
            tqdm.write(f"Saved model in epoch {epoch}")

    fig, ax = plot_loss_curve(train_loss_list, val_loss_list)
    fig.savefig(result_folder / "loss_curve.png", dpi=200)
    