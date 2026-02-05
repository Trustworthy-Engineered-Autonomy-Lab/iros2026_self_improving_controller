import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from utils import load_data, construct_data, plot_loss_curve, EarlyStopCriterion

from tqdm import tqdm

BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100
RESULT_FOLDER = "critic_%Y_%m_%d_%H_%M_%S"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class CriticDataset(Dataset):
    RAMP_TURN_POINT = 0.2
    def __init__(self, images, steers, expand_times = 1, label_scheme = 'binary', **kwargs):
        super().__init__()
        self._images = images
        rand_steers = torch.rand(steers.shape[0] * expand_times, 1, device=steers.device)
        real_steers = steers.repeat(expand_times + 1, *([1] * (steers.dim() - 1)))
        self._steers = torch.concatenate([steers, rand_steers])
        self._labels = self._make_label(real_steers, self._steers, label_scheme, **kwargs)

    def _make_label(self, steer, real_steer, label_scheme = 'binary', **kwargs):
        if label_scheme == 'binary':
            label = (steer == real_steer).float()
        elif label_scheme == 'ramp':
            diff = torch.abs(steer - real_steer)
            pos_idxs = torch.where(diff <= kwargs.get('ramp_turn_point', CriticDataset.RAMP_TURN_POINT))[0]
            label = torch.zeros_like(steer)
            label[pos_idxs] = 1 - diff[pos_idxs]
        else:
            raise ValueError(f"Unsupported label scheme {self._label_scheme}")

        return label

    def __len__(self):
        return self._labels.shape[0]
    
    def __getitem__(self, index):

        real_len = self._images.shape[0]
        image = self._images[index % real_len]
        steer = self._steers[index]
        label = self._labels[index]
        
        return image, steer, label


class Critic(nn.Module):
    """
    CNN using kernel sizes from the diagram:
    5x5, 5x5, 5x5, 3x3, 3x3
    No image resizing.
    Input: (N, 3, 144, 224)
    Output: (N, 1)
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        # For input 3x144x224 → conv stack → 64x11x21
        self.flatten_dim = 64 * 11 * 21 + 1

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 50),
            nn.ReLU(inplace=True),

            nn.Linear(50, 10),
            nn.ReLU(inplace=True),

            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # Normalize to [-1,1] as in NVIDIA paper
        x1 = x1.permute(0,3,1,2).float()
        x1 = x1 / 255.0
        x1 = x1 * 2.0 - 1.0

        x1 = self.features(x1)
        x1 = x1.reshape(x1.size(0), -1)
        x = torch.cat([x1, x2], dim=1)
        return self.classifier(x)
    
def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer
    ):
    model.train()
    total_loss = 0
    for image, steer, label in dataloader:
        pred = model(image, steer)
        loss = criterion(pred, label).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def val_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module
    ):

    model.eval()
    total_loss = 0
    for image, steer, label in dataloader:
        with torch.no_grad():
            pred = model(image, steer)
            loss = criterion(pred, label).mean()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss
    
def train(
        model: nn.Module,
        images: torch.Tensor,
        steers: torch.Tensor,
        device = DEVICE,
        batch_size = BATCH_SIZE,
        nepochs = NUM_EPOCHS,
        lr = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
        dataset_config = {}
    ):

    device = torch.device(device)

    images = images.to(device, torch.float32)
    steers = steers.to(device, torch.float32)

    model = model.to(device)

    dataset = CriticDataset(images, steers, **dataset_config)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    criterion = nn.MSELoss(reduction='none')  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = range(1, nepochs+1)

    pbar = tqdm(epochs, desc="Training critic")

    for epoch in pbar:
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer)
        val_loss = val_epoch(model, val_dataloader, criterion)

        pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

        yield train_loss, val_loss
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('config', type=str)

    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Failed to load config file {args.config}: {e}")
        sys.exit(1)

    result_folder = Path(datetime.now().strftime(config.get('result_folder', RESULT_FOLDER)))

    try:
        data = load_data(config['load_data'])

    except Exception as e:
        print(f"Failed to load training data: {e}")
        sys.exit(1)

    try:
        training_data = construct_data(data, config.get('training_data', {}))
    except Exception as e:
        print(f"Failed to construct training data: {e}")
        sys.exit(1)

    images = torch.from_numpy(training_data['image'])
    steers = torch.from_numpy(training_data['steer']).unsqueeze(1)

    model = Critic()
    
    result_folder.mkdir(parents=True, exist_ok=True)
    
    train_loss_list = []
    val_loss_list = []

    train_config = config.get('train',{})
    early_stop = EarlyStopCriterion()

    for epoch, (train_loss, val_loss) in enumerate(train(
        model,
        images,
        steers,
        **train_config
    ),1):

        if early_stop(train_loss, val_loss):
            torch.save(model.state_dict(), result_folder / 'best.pt')
            tqdm.write('Model saved')

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

    fig, ax = plot_loss_curve(train_loss_list, val_loss_list)
    fig.savefig(result_folder / "loss_curve.png", dpi=200)

