import sys
from pathlib import Path
import pickle

import argparse

from ..utils import proc_collected_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('collected_data_path', type=str)
    parser.add_argument('output_data_path', type=str)

    args = parser.parse_args()

    try:
        data = proc_collected_data(args.collected_data_path)
    except Exception as e:
        print(f"Failed to load and process collected data under {args.collected_data_path}: {e}")
        sys.exit(1)

    try:
        output_data_path = Path(args.output_data_path)
        output_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_data_path, 'wb') as f:
            pickle.dump(data, f)

    except Exception as e:
        print(f"Failed to write processed data {output_data_path}: {e}")
        sys.exit(1)

    print(f"Processed data is saved to {output_data_path}")

    

