import argparse
from pathlib import Path
import sys
import json
import pickle

import torch
import torch.nn.functional as F
import numpy as np

from utils import proc_collected_data
from tools.pt2onnx import export_onnx

import cae
from cae import CAE, normalize_image
import critic
from critic import Critic
import cnn_controller
from cnn_controller import CNN
import cae_steer
from cae_steer import CAESteer

from utils import EarlyStopCriterion

from datetime import datetime
from tqdm import tqdm


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RC_THRESHOLD = 0.5
CRITIC_THRESHOLD = 0.5
RESULT_FOLDER = 'evolve_%Y_%m_%d_%H_%M_%S'


def median_filter_1d(x, kernel_size):
    pad = kernel_size // 2
    x_padded = F.pad(x.unsqueeze(0).unsqueeze(0), (pad, pad), mode='reflect')
    windows = x_padded.unfold(dimension=2, size=kernel_size, step=1)
    return windows.median(dim=-1).values.squeeze()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('config', type=str)

    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Failed to load config from {args.config}: {e}")
        sys.exit(1)

    image_list = []
    steer_list = []
    try:
        data_config = config['data']
        for k,v in data_config.items():
            data_path = Path(k)
            print(f"Loading {data_path}")
            if data_path.is_dir():
                data = proc_collected_data(data_path)
            elif data_path.suffix == '.pkl':
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                print(f"Unrecognized data format {data_path.suffix}. Ignored {data_path}")
                continue
                
            image_list.append(data['image'][slice(*v)])
            steer_list.append(data['steer'][slice(*v)].reshape(-1,1))
    except Exception as e:
        print(f"Failed to process collected data under {config['collected_data']}: {e}")
        sys.exit(1)

    DEVICE = config.get('device', DEVICE)
    try:
        device = torch.device(DEVICE)
    except Exception as e:
        print(f"Invalid device {DEVICE}: {e}")
        sys.exit(1)

    result_folder = Path(datetime.now().strftime(config.get('result_folder', RESULT_FOLDER)))
    result_folder.mkdir(parents=True, exist_ok=True)

    images = torch.from_numpy(np.concatenate(image_list)).to(device, torch.float32)
    steers = torch.from_numpy(np.concatenate(steer_list)).to(device, torch.float32)

    print(f"Loaded {images.shape[0]} image - steer pairs")

    #-----------------------------------------------

    cae_steer_config = config.get('cae_steer', {})
    cae_steer_train_config = cae_steer_config.get('train', {})
    cae_steer_eval_config = cae_steer_config.get('eval',  {})
    cae_steer_model = CAESteer().to(device)

    normalized_images = normalize_image(images).permute(0,3,1,2)

    for loss in cae_steer.train(
        cae_steer_model,
        normalized_images,
        steers,
        device = device,
        **cae_steer_train_config
    ):
        pass

    mse = cae_steer.eval(
        cae_steer_model, 
        normalized_images, 
        steers, 
        device = device, 
        **cae_steer_eval_config
    )

    # mse = median_filter_1d(mse, 11)

    mse_threshold = cae_steer_config.get('mse_threshold', RC_THRESHOLD)
    if isinstance(mse_threshold, str):
        mse_threshold = eval(mse_threshold, {}, {'mean' : torch.mean(mse), 'stdev': torch.std(mse)})
        print(f"MSE threshold is {mse_threshold:.4f}")
    
    selected = torch.where(mse <= mse_threshold)[0]

    n_removed = images.shape[0] - len(selected)

    images = images[selected]
    steers = steers[selected]

    print(f"Removed {n_removed} images whose mse > {mse_threshold:.4f}")

    with open(result_folder / 'cleaned_data.pkl', 'wb') as f:
        pickle.dump({
            'image' : images.detach().cpu().numpy(),
            'steer': steers.detach().cpu().numpy().reshape(-1)
        }, f)

    print(f"Saved cleaned data {result_folder / 'cleaned_data.pkl'}")
        
    #-----------------------------------------------

    cnn_model = CNN().to(device)

    cnn_config = config.get('cnn', {})
    cnn_train_config = cnn_config.get('train', {})

    early_stop = EarlyStopCriterion()

    best_model = cnn_model

    for train_loss, val_loss in cnn_controller.train(
        cnn_model,
        images,
        steers,
        device,
        **cnn_train_config
    ):
        if early_stop(train_loss, val_loss):
            best_model = cnn_model

    cnn_export_config = cnn_config.get('export', {})
    
    torch.save(cnn_model.state_dict(), result_folder/ 'cnn_controller.pt')
    export_onnx(cnn_model, images[:1], result_folder/ 'cnn_controller.onnx', **cnn_export_config)

    print(f"CNN Controller Models are saved to {result_folder}")
    
    

