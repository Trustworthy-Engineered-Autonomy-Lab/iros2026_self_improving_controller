import argparse
from pathlib import Path
import sys
import json
import pickle

import torch
import numpy as np

from utils import proc_collected_data
from tools.pt2onnx import export_onnx

import cae
from cae import CAE, normalize_image
import critic
from critic import Critic
import cnn_controller
from cnn_controller import CNN

from utils import EarlyStopCriterion

from datetime import datetime


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RC_THRESHOLD = 0.5
CRITIC_THRESHOLD = 0.5
RESULT_FOLDER = 'evolve_%Y_%m_%d_%H_%M_%S'

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

    cae_config = config.get('cae', {})
    cae_train_config = cae_config.get('train', {})
    cae_model = CAE().to(device)

    normalized_images = normalize_image(images).permute(0,3,1,2)

    for loss in cae.train(
        cae_model,
        normalized_images,
        device,
        **cae_train_config
    ):
        pass

    rc, mse = cae.eval(cae_model, normalized_images, device)

    rc_threshold = cae_config.get('rc_threshold', RC_THRESHOLD)
    if isinstance(rc_threshold, str):
        rc_threshold = eval(rc_threshold, {}, {'mean' : torch.mean(rc), 'stdev': torch.std(rc)})
        print(f"RC threshold is {rc_threshold:.2f}")
    
    selected = torch.where(rc >= rc_threshold)[0]

    n_removed = images.shape[0] - len(selected)

    images = images[selected]
    steers = steers[selected]

    print(f"Removed {n_removed} images whose rc value < {rc_threshold:.2f}")

    #-----------------------------------------------

    critic_config = config.get('critic', {})
    critic_train_config = critic_config.get('train', {})

    critic_model = Critic().to(device)

    for loss in critic.train(
        critic_model,
        images,
        steers,
        device,
        **critic_train_config
    ):
        pass
    
    critic_model.eval()
    critic_score = critic_model(images, steers)

    critic_threshold = critic_config.get('threshold', CRITIC_THRESHOLD)
    selected = torch.where(critic_score >= critic_threshold)[0]

    n_removed = images.shape[0] - len(selected)

    images = images[selected]
    steers = steers[selected]

    print(f"Removed {n_removed} images whose critic score < {critic_threshold:.2f}")

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
    
    

