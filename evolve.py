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
    # Step 1: CAE-Steer - compute MSE for confidence weighting

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

    print(f"CAE-Steer MSE: min={mse.min().item():.4f}, max={mse.max().item():.4f}, mean={mse.mean().item():.4f}")

    #-----------------------------------------------
    # Step 2: CAE - compute PCC (RC) for confidence weighting

    cae_config = config.get('cae', {})
    cae_train_config = cae_config.get('train', {})

    cae_model = CAE().to(device)

    normalized_images = normalize_image(images).permute(0,3,1,2)

    for loss in cae.train(cae_model, normalized_images, device, **cae_train_config):
        pass

    pcc, _ = cae.eval(cae_model, normalized_images, device)

    print(f"CAE PCC: min={pcc.min().item():.4f}, max={pcc.max().item():.4f}, mean={pcc.mean().item():.4f}")

    #-----------------------------------------------
    # Step 3: Quadrant classification + steer repair

    mse_median = torch.median(mse)
    pcc_median = torch.median(pcc)

    # Get CAE-Steer predicted steers for repair
    pred_steers = cae_steer.predict(cae_steer_model, normalized_images, device)

    # Quadrant masks
    mask_cat3 = (mse < mse_median) & (pcc > pcc_median)  # good action + normal image
    mask_cat1 = (mse < mse_median) & (pcc < pcc_median)  # good action + abnormal image
    mask_cat4 = (mse > mse_median) & (pcc > pcc_median)  # bad action + normal image → repair
    mask_cat2 = (mse > mse_median) & (pcc < pcc_median)  # bad action + abnormal image → discard

    # Repair Cat 4 steers with CAE-Steer predictions
    cleaned_steers = steers.clone()
    cleaned_steers[mask_cat4] = pred_steers[mask_cat4]

    # Assign weights
    sample_weights = torch.zeros_like(mse)
    sample_weights[mask_cat3] = 1.0  # best data
    sample_weights[mask_cat1] = 0.5  # error-correction
    sample_weights[mask_cat4] = 0.5  # repaired
    # Cat 2: weight = 0 (default)

    print(f"MSE median={mse_median.item():.4f}, PCC median={pcc_median.item():.4f}")
    print(f"Cat 3 (good action + normal image):   {mask_cat3.sum().item()} samples, weight=1.0")
    print(f"Cat 1 (good action + abnormal image): {mask_cat1.sum().item()} samples, weight=0.5")
    print(f"Cat 4 (repaired steer + normal image): {mask_cat4.sum().item()} samples, weight=0.5")
    print(f"Cat 2 (discarded):                    {mask_cat2.sum().item()} samples, weight=0")
    print(f"Sample weights: min={sample_weights.min().item():.4f}, max={sample_weights.max().item():.4f}, mean={sample_weights.mean().item():.4f}")

    #-----------------------------------------------
    with open(result_folder / 'confidence_data.pkl', 'wb') as f:
        pickle.dump({
            'image': images.detach().cpu().numpy(),
            'steer': steers.detach().cpu().numpy().reshape(-1),
            'cleaned_steer': cleaned_steers.detach().cpu().numpy().reshape(-1),
            'mse': mse.detach().cpu().numpy(),
            'pcc': pcc.detach().cpu().numpy(),
            'quadrant': np.array(['cat3' if m3 else 'cat1' if m1 else 'cat4' if m4 else 'cat2'
                                  for m3, m1, m4 in zip(mask_cat3.cpu(), mask_cat1.cpu(), mask_cat4.cpu())]),
            'weights': sample_weights.detach().cpu().numpy()
        }, f)

    print(f"Saved confidence data {result_folder / 'confidence_data.pkl'}")

    #-----------------------------------------------

    cnn_model = CNN().to(device)

    cnn_config = config.get('cnn', {})
    cnn_train_config = cnn_config.get('train', {})

    early_stop = EarlyStopCriterion()

    best_model = cnn_model

    for train_loss, val_loss in cnn_controller.train(
        cnn_model,
        images,
        cleaned_steers,
        device,
        weights=sample_weights,
        **cnn_train_config
    ):
        if early_stop(train_loss, val_loss):
            best_model = cnn_model

    cnn_export_config = cnn_config.get('export', {})
    
    torch.save(cnn_model.state_dict(), result_folder/ 'cnn_controller.pt')
    export_onnx(cnn_model, images[:1], result_folder/ 'cnn_controller.onnx', **cnn_export_config)

    print(f"CNN Controller Models are saved to {result_folder}")
    
    

