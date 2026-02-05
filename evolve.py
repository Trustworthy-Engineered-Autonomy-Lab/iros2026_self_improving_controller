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
    # Step 3: Compute confidence weights from MSE and PCC

    # ===== Three-Stage MSE Confidence Calculation =====
    # User's design: Leverage MSE distribution characteristic (median < mean for right-skewed)
    # Stage 1 (MSE < median): Linear growth to 0.7
    # Stage 2 (median ≤ MSE < mean): Steep decay to 0.4
    # Stage 3 (MSE ≥ mean): Exponential decay to ~0

    # Step 1: Calculate distribution landmarks
    mse_median = torch.median(mse)
    mse_mean = mse.mean()
    mse_min = mse.min()

    # Step 2: Three-stage MSE confidence
    conf_stage1 = 1.0 

    # Stage 2: Degrading action (median ≤ MSE < mean) → quadratic decay 0.7 to 0.4
    ratio_stage2 = (mse - mse_median) / (mse_mean - mse_median + 1e-8)
    conf_stage2 = 1.0 - 0.2 * (ratio_stage2 ** 2)
    conf_stage2 = conf_stage2.clamp(0.8, 1.0)

    # Stage 3: Poor action (MSE ≥ mean) → exponential decay 0.4 to 0
    decay_rate = 2.0  # Tunable: higher = harsher punishment
    conf_stage3 = 0.8 * torch.exp(-decay_rate * (mse - mse_mean) / (mse_mean - mse_median + 1e-8))
    conf_stage3 = conf_stage3.clamp(0.0, 0.8)

    # Combine three stages
    mse_confidence = torch.where(
        mse < mse_median,
        conf_stage1,
        torch.where(
            mse < mse_mean,
            conf_stage2,
            conf_stage3
        )
    )

    # Step 3: PCC confidence with lower bound
    # Even for poor quality images, maintain minimum weight for error correction learning
    pcc_median = torch.median(pcc)
    pcc_mad = torch.median(torch.abs(pcc - pcc_median))
    pcc_z = (pcc - pcc_median) / (pcc_mad + 1e-8)

    pcc_lower_bound = 0.4  # Tunable: minimum weight for poor quality images (0.4-0.6)

    # Map pcc_z to [pcc_lower_bound, 1.0]
    # pcc_z > 0 (good quality) → confidence approaches 1.0
    # pcc_z < -2 (poor quality) → confidence approaches pcc_lower_bound
    pcc_confidence = torch.where(
        pcc_z > 0,
        # Good quality: linear increase to 1.0
        torch.clamp(pcc_lower_bound + (1.0 - pcc_lower_bound) * (1.0 + 0.5 * pcc_z), pcc_lower_bound, 1.0),
        # Poor quality: exponential decay to lower_bound
        pcc_lower_bound + (1.0 - pcc_lower_bound) * torch.exp(0.5 * pcc_z)
    )
    pcc_confidence = torch.clamp(pcc_confidence, pcc_lower_bound, 1.0)

    # Step 4: Combine using squared MSE and PCC
    # final_weight = (mse_confidence)^2 * pcc_confidence
    # Squaring MSE amplifies differences between good/bad actions
    # PCC provides bounded adjustment [pcc_lower_bound, 1.0]
    sample_weights = (mse_confidence ** 2) * pcc_confidence
    sample_weights = torch.clamp(sample_weights, 0.0, 1.0)

    print(f"MSE confidence: min={mse_confidence.min().item():.4f}, max={mse_confidence.max().item():.4f}, mean={mse_confidence.mean().item():.4f}")
    print(f"PCC confidence: min={pcc_confidence.min().item():.4f}, max={pcc_confidence.max().item():.4f}, mean={pcc_confidence.mean().item():.4f}")
    print(f"Sample weights: min={sample_weights.min().item():.4f}, max={sample_weights.max().item():.4f}, mean={sample_weights.mean().item():.4f}")

    #-----------------------------------------------
    with open(result_folder / 'confidence_data.pkl', 'wb') as f:
        pickle.dump({
            'image': images.detach().cpu().numpy(),
            'steer': steers.detach().cpu().numpy().reshape(-1),
            'mse': mse.detach().cpu().numpy(),
            'pcc': pcc.detach().cpu().numpy(),
            'mse_confidence': mse_confidence.detach().cpu().numpy(),
            'pcc_confidence': pcc_confidence.detach().cpu().numpy(),
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
        steers,
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
    
    

