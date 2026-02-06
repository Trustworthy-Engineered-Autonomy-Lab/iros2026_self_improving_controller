import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import argparse
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("result_folder")
    parser.add_argument("--step", type=int, default=1)

    args = parser.parse_args()

    result_folder = Path(args.result_folder)

    if not result_folder.exists():
        print("Result folder does not exist")
        sys.exit(1)

    try:
        with open(result_folder / "run.json", 'r') as f:
            config = json.load(f)

        lambda_list = config['lambda_list']
        batch_size = config['batch_size']
        epochs = config['record_epochs']

    except Exception as e:
        print(f"Result is incomplete : {e}")
        sys.exit(1)

    for gamma_value in lambda_list:
        # 构建结果目录路径
        results_dir = result_folder/f"lambda_{gamma_value}"

        # 检查目录是否存在
        if not os.path.exists(results_dir):
            print(f"结果目录 {results_dir} 不存在，跳过 lambda={gamma_value}")
            continue

        # 读取 rc_records.npy
        rc_records_path = os.path.join(results_dir, 'rc_records.npy')
        if not os.path.exists(rc_records_path):
            print(f"文件 {rc_records_path} 不存在，跳过 lambda={gamma_value}")
            continue
        rc_records = np.load(rc_records_path, allow_pickle=True).item()

        # 读取 smoothed_rc_records.npy
        smoothed_rc_records_path = os.path.join(results_dir, 'smoothed_rc_records.npy')
        if not os.path.exists(smoothed_rc_records_path):
            print(f"文件 {smoothed_rc_records_path} 不存在，跳过 lambda={gamma_value}")
            continue
        smoothed_rc_records = np.load(smoothed_rc_records_path, allow_pickle=True).item()

        # 创建图形和子图
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))  # 两个子图，尺寸为 12x10
        colors = ['b', 'g', 'r', 'c', 'm']  # 定义颜色列表

        for idx, epoch in enumerate(epochs[::args.step]):
            # 获取指定 epoch 的 RC 值和索引
            if epoch in rc_records:
                rc_values = rc_records[epoch]
                smoothed_rc_values = smoothed_rc_records[epoch]
            else:
                print(f"在 lambda={gamma_value} 下，epoch={epoch} 的 RC 记录不存在，跳过")
                continue

            color = colors[idx % len(colors)]  # 循环使用颜色
            
            # 绘制 RC 值（上面的子图）
            axes[0].plot(range(len(rc_values)), rc_values, label=f'Epoch {epoch}', color=color, alpha=0.6)

            # 绘制平滑后的 RC 值（下面的子图）
            axes[1].plot(range(len(smoothed_rc_values)), smoothed_rc_values, label=f'Epoch {epoch}', color=color, alpha=0.8)

        # 设置上面子图的属性
        axes[0].set_xlabel('Image Index')
        axes[0].set_ylabel('RC Value')
        axes[0].set_title(f'RC Values (Lambda={gamma_value})')
        axes[0].legend()
        axes[0].grid(True)

        # 设置下面子图的属性
        axes[1].set_xlabel('Image Index')
        axes[1].set_ylabel('Smoothed RC Value')
        axes[1].set_title(f'Smoothed RC Values (Lambda={gamma_value})')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()

        # 保存图像
        save_dir = Path(result_folder / "RC_smooth_plots")
        save_dir.mkdir(parents=False, exist_ok=True)
        plot_filename = save_dir / f'rc_lambda_{gamma_value}_epochs.png'
        plt.savefig(plot_filename, dpi=300)  # dpi 可以根据需要调整
        print(f"RC 曲线图已保存为：{plot_filename}")

        # 关闭图像，释放内存
        plt.close()