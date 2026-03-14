#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练指标可视化

解析训练日志（TensorBoard、wandb、自定义 log），绘制 loss/reward 曲线。
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    log_dir: str | Path,
    metrics: list[str] | None = None,
    output_path: str | Path | None = None,
) -> None:
    """
    从 TensorBoard 或简单 CSV 日志绘制训练曲线

    Args:
        log_dir: 日志目录（含 events.out.tfevents 或 metrics.csv）
        metrics: 要绘制的指标名，None 则尝试自动检测
        output_path: 保存图片路径
    """
    log_dir = Path(log_dir)
    if not log_dir.exists():
        raise FileNotFoundError(f"Log dir not found: {log_dir}")

    # 尝试加载 TensorBoard
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        event_files = list(log_dir.glob("events.out.tfevents*"))
        if event_files:
            ea = EventAccumulator(str(event_files[0]))
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            if metrics is None:
                metrics = tags[:8]  # 默认取前 8 个
            n = len(metrics)
            if n == 0:
                print("No scalar metrics found")
                return
            fig, axes = plt.subplots((n + 1) // 2, min(2, n), figsize=(10, 4 * ((n + 1) // 2)))
            if n == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            for i, tag in enumerate(metrics):
                if tag in tags:
                    events = ea.Scalars(tag)
                    steps = [e.step for e in events]
                    vals = [e.value for e in events]
                    axes[i].plot(steps, vals)
                    axes[i].set_title(tag)
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            if output_path:
                plt.savefig(output_path, dpi=150)
            plt.show()
            return
    except ImportError:
        pass

    # 回退：简单 CSV 格式
    csv_path = log_dir / "metrics.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        cols = [c for c in df.columns if c != "step" and c != "epoch"]
        if metrics:
            cols = [c for c in cols if c in metrics]
        n = len(cols)
        if n == 0:
            print("No metrics columns found")
            return
        fig, axes = plt.subplots((n + 1) // 2, min(2, n), figsize=(10, 4 * ((n + 1) // 2)))
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        x = df.get("step", df.get("epoch", np.arange(len(df))))
        for i, col in enumerate(cols):
            axes[i].plot(x, df[col])
            axes[i].set_title(col)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150)
        plt.show()
    else:
        print(f"No TensorBoard events or metrics.csv found in {log_dir}")


def plot_custom_curves(
    data: dict[str, list[float]],
    x_label: str = "step",
    output_path: str | Path | None = None,
) -> None:
    """
    绘制自定义数据曲线

    Args:
        data: {"metric_name": [values], "step": [0,1,2,...]} 或 {"metric": values}
        x_label: x 轴对应的 key
        output_path: 保存路径
    """
    steps = data.get(x_label, np.arange(max(len(v) for v in data.values() if hasattr(v, "__len__"))))
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, v in data.items():
        if k == x_label:
            continue
        if hasattr(v, "__len__"):
            ax.plot(steps[: len(v)], v, label=k)
    ax.legend()
    ax.set_xlabel(x_label)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()
