import pickle
from pathlib import Path
import pandas
import cv2
import numpy as np

from torch.utils.data import TensorDataset
import torch

import matplotlib.pyplot as plt

def normalize_image(img: torch.Tensor):
    x_min = img.min()
    x_max = img.max()
    img_norm = (img - x_min) / (x_max - x_min + 1e-8) 
    return img_norm.to(torch.float32)

def proc_collected_data(data_folder):
    data_path = Path(data_folder)
    csv = pandas.read_csv(data_path / 'labels.csv')
    image_list = []
    steer_list = []
    throttle_list = []
    for line in csv.itertuples():
        idx, filename, steer, throttle = line
        image = cv2.imread(str(data_path / 'images' / filename))
        image_list.append(image)
        steer_list.append(steer)
        throttle_list.append(throttle)

    return {
        "image" : np.array(image_list),
        "steer" : np.array(steer_list),
        "throttle" : np.array(throttle_list)
    }

def load_data(config: dict):
    data = {}
    for k,v in config.items():
        path = Path(v)
        if path.is_dir():
                data[k] = proc_collected_data(path)
        elif path.suffix == '.pkl':
            with open(path, 'rb') as f:
                data[k] = pickle.load(f)
        else:
            print(f"Unrecognized data format {path.suffix}. Ignored {path}")
            continue

        print(f"Loading training data {v}")
    return data

def construct_data(data: dict, config: dict):
    image_list = []
    steer_list = []
    for k,v in config.items():
        print(f"Read data {k}")
        image_list.append(data[k]['image'][slice(*v)])
        steer_list.append(data[k]['steer'][slice(*v)])
    return {
        'image' : np.concatenate(image_list),
        'steer' : np.concatenate(steer_list)
    }

def plot_loss_curve(train_loss, val_loss):

    fig, ax = plt.subplots()

    ax.plot(range(len(train_loss)), train_loss, '-*', label="Train")
    ax.plot(range(len(val_loss)), val_loss, '-.', label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    return fig, ax
    

class EarlyStopCriterion:
    def __init__(self):
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')

    def __call__(self, train_loss, val_loss, *args, **kwds):
        save_model = 0
        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss
            save_model += 1

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            save_model += 1

        return save_model == 2
    
# 评估数据集和数据加载器（不打乱顺序，并返回索引）
class IndexedTensorDataset(TensorDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return (*data, idx)
