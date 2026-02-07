import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import pickle

from utils import load_data, construct_data, normalize_image, plot_loss_curve, EarlyStopCriterion
from cae import ImageEncoder, ImageDecoder

class SteerDecoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )

class SteerEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
    
class CAESteer(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = ImageEncoder()
        self.steer_encoder = SteerEncoder()
        self.decoder = SteerDecoder()

    def forward(self, image, steer=None):
        latent_image = self.encoder(image)
        pred_steer = self.decoder(latent_image)
        latent_steer = self.steer_encoder(steer) if steer is not None else None

        return pred_steer, latent_image, latent_steer

LEARNING_RATE = 5e-4
BATCH_SIZE = 256  
WEIGHT_DECAY = 5e-4

NUM_EPOCHS = 100
RECORD_INTERVAL = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULT_FOLDER = "cae_%Y_%m_%d_%H_%M_%S"
LAMBDA = 1
    
def eval(
        model: nn.Module,
        images : torch.Tensor,
        steers : torch.Tensor,
        device = DEVICE,
        batch_size = BATCH_SIZE
    ):

    images = images.to(device, torch.float32)
    steers = steers.to(device, torch.float32)

    dataset = TensorDataset(images, steers)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss(reduction='none')
    
    model.eval()
    loss_list = []
    with torch.no_grad():
        for image, steer in val_loader:
            pred_steer, _, _ = model(image)
            loss = criterion(pred_steer, steer).mean(dim=1)
            loss_list.append(loss)

    return torch.concatenate(loss_list)

def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        lam: float = LAMBDA,
    ):
    model.train()
    total_loss = 0
    for image, steer in dataloader:

        pred_steer, latent_image, latent_steer = model(image, steer)
        mse_loss = criterion(pred_steer, steer)
        align_loss = F.mse_loss(latent_image, latent_steer)
        loss = mse_loss + lam * align_loss

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
        weight_decay = WEIGHT_DECAY,
        lam = LAMBDA,
    ):
    images = images.to(device, torch.float32)
    steers = steers.to(device, torch.float32)
    model = model.to(device)

    train_dataset = TensorDataset(images, steers)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    pbar = tqdm(range(1, nepochs + 1), desc=f"Training CAE steer")
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, criterion, optimizer, lam)

        pbar.set_postfix(train_loss=f"{train_loss:.4f}")

        yield train_loss

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
    mse_record = {}
    for epoch, train_loss in enumerate(train(model, images, steers, **train_config), 1):
        train_loss_list.append(train_loss)
        
        if epoch % record_interval == 0:
            mse_record[epoch] = eval(model, images, steers, **eval_config).detach().cpu().numpy()
            tqdm.write("MSE recorded")

    if epoch not in mse_record:
        mse_record[epoch] = eval(model, images, steers, **eval_config)

    with open(result_folder / 'mse_record.pkl', 'wb') as f:
        pickle.dump(mse_record, f)

    fig, ax = plot_loss_curve(train_loss_list)
    fig.savefig(result_folder / "loss_curve.png", dpi=200)
    torch.save(model.state_dict(), result_folder / 'model.pt')
    