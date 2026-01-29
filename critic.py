import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from utils import load_data, construct_data, EarlyStopCriterion

from tqdm import tqdm

BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100
RESULT_FOLDER = "critic_%Y_%m_%d_%H_%M_%S"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        labels: torch.Tensor,
        device = DEVICE,
        batch_size = BATCH_SIZE,
        nepochs = NUM_EPOCHS,
        lr = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
    ):

    device = torch.device(device)

    images = images.to(device, torch.float32)
    steers = steers.to(device, torch.float32)
    labels = labels.to(device, torch.float32)

    model = model.to(device)

    dataset = TensorDataset(images, steers, labels)
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

def _generate_data(data, source, times = 1, method = 'uniform'):
    images = data[source]['image']
    steers = data[source]['steer']

    gen_images = np.broadcast_to(images, (times * images.shape[0], *images.shape[1:]))
    if method == 'uniform':
        gen_steers = np.random.rand(times * steers.shape[0]) * 2 -1
    else:
        raise ValueError(f"Unsupported method {method}")
    
    return {
        'image': gen_images,
        'steer': gen_steers
    }

def generate_data(data: dict, config: dict):
    gen_data = {}
    for k,v in config.items():
        gen_data[k] = _generate_data(data, **v)
    return gen_data

def construct_label(positive_data: dict, negitive_data: dict, scheme = 'binary'):
    if scheme == 'binary':
        labels = np.concatenate([np.ones_like(positive_data['steer']), np.zeros_like(negitive_data['steer'])])
    else:
        raise ValueError(f"Unknown label scheme {scheme}")
    
    return labels
        
def construct_critic_data(data, config: dict):
    data_config = config.get('data', {})
    positive_config = data_config.get('positive', {})
    negative_config = data_config.get('negative', {})

    print("Constructing positive data")
    positive_data = construct_data(data, positive_config)
    print("Constructing negative data")
    negative_data = construct_data(data, negative_config)

    label_config = config.get('label', {})
    print("Constructing labels")
    labels = construct_label(positive_data, negative_data, **label_config)

    return {
        'image' : np.concatenate([positive_data['image'], negative_data['image']]),
        'steer' : np.concatenate([positive_data['steer'], negative_data['steer']]),
        'label' : labels
    }
    
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
        data = data | generate_data(data, config.get('generate_data', {}))
    except Exception as e:
        print(f"Failed to generate training data: {e}")
        sys.exit(1)

    try:
        training_data = construct_critic_data(data, config['training_data'])
        images = torch.from_numpy(training_data['image'])
        steers = torch.from_numpy(training_data['steer']).unsqueeze(1)
        labels = torch.from_numpy(training_data['label']).unsqueeze(1)

    except Exception as e:
        print(f"Failed to construct training data: {e}")
        sys.exit(1)

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
        labels,
        **train_config
    ),1):

        if early_stop(train_loss, val_loss):
            torch.save(model.state_dict(), result_folder / 'best.pt')
            tqdm.write('Model saved')

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

    fig, ax = plt.subplots()

    ax.plot(range(len(train_loss_list)), train_loss_list, '-*', label="Train")
    ax.plot(range(len(val_loss_list)), val_loss_list, '-.', label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(result_folder / "loss_curve.png", dpi=200)
    plt.close(fig)   # <-- important to prevent memory buildup
        


    


