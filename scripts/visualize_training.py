#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练曲线可视化脚本

用法:
    python scripts/visualize_training.py outputs/run1
    python scripts/visualize_training.py outputs/run1 --output plot.png
"""

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualization.metrics import plot_training_curves


def main():
    parser = argparse.ArgumentParser(description="绘制训练曲线")
    parser.add_argument("log_dir", type=str, help="日志目录（TensorBoard 或含 metrics.csv）")
    parser.add_argument("--output", type=str, default=None, help="保存图片路径")
    parser.add_argument("--metrics", type=str, nargs="*", help="指定要绘制的指标名")
    args = parser.parse_args()

    plot_training_curves(
        args.log_dir,
        metrics=args.metrics,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
