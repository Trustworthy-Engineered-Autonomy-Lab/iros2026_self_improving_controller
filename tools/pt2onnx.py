import torch
from cnn_controller import CNN
from pathlib import Path
import argparse
import sys


def export_onnx(model, test_input, save_path, input_name = "image", output_name = "steer"):
    torch.onnx.export(
        model,
        test_input,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=[input_name],
        output_names=[output_name]
    )


def _parse_tuple(s):
    return tuple(map(int, s.split(',')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("model_path", type=str)
    parser.add_argument("--input_name", type=str, default="image")
    parser.add_argument("--output_name", type=str, default="steer")
    parser.add_argument('--shape', type=_parse_tuple, default=(1, 144, 224, 3))

    args = parser.parse_args()

    model_path = Path(args.model_path)

    model = CNN()
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except Exception as e:
        print(f'Failed to load model {args.model}: {e}')
        sys.exit(1)

    export_onnx(model, torch.randn(*args.shape), model_path.with_suffix('.onnx'), args.input_name, args.output_name)
