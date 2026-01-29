import torch

import argparse
from pathlib import Path
from critic import Critic
import sys
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("model_path", type=str)
    parser.add_argument("data_path", type=str)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    try:
        device = torch.device(args.device)
    except Exception as e:
        print(f"Invalid device {args.device}: {e}")
        sys.exit(1)

    model = Critic().to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
    except Exception as e:
        print(f"Failed to load model {args.model_path}: {e}")
        sys.exit(1)

    try:
        data_path = Path(args.data_path)

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        images = torch.from_numpy(data['images']).to(device, dtype=torch.float32)
        steers = torch.from_numpy(data['steer']).unsqueeze().to(device, dtype=torch.float32)

    except Exception as e:
        print(f"Failed to load data {args.data_path}: {e}")
        sys.exit(1)

    preds = model(images, steers)

    idxs = torch.where(preds > args.threshold)
    cleaned_images = images[idxs]
    cleaned_steers = steers[idxs]

    cleaned_data = {
        "images" : cleaned_images.detach().cpu().numpy(),
        "steer" : cleaned_steers.detach().cpu().numpy()
    }

    cleaned_data_path = data_path.with_name(f"{data_path.name}_cleaned")
    with open(cleaned_data_path, 'wb') as f:
        pickle.dump(cleaned_data, f)

    print(f"Cleaned data saved as {cleaned_data_path}")
    
