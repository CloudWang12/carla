from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from carla_ml.config import (
    ARTIFACTS_DIR,
    DATASET_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    TARGET_COLUMNS,
    FEATURE_COLUMNS,
)
from carla_ml.data import build_sequences
from carla_ml.model import DrivingLSTM


class WeightedTrajectoryLoss(nn.Module):
    def __init__(self, target_mean: torch.Tensor, target_std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("target_mean", target_mean.view(1, 1, -1))
        self.register_buffer("target_std", target_std.view(1, 1, -1))

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        control_loss = torch.mean((preds[..., :2] - targets[..., :2]) ** 2)
        planar_loss = torch.mean((preds[..., 2:4] - targets[..., 2:4]) ** 2)
        dz_loss = torch.mean((preds[..., 4:5] - targets[..., 4:5]) ** 2)
        rot_loss = torch.mean((preds[..., 5:11] - targets[..., 5:11]) ** 2)
        return control_loss + 2.0 * planar_loss + dz_loss + 2.5 * rot_loss


def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> float:
    # 验证阶段只做前向计算，不会更新模型参数。
    # loader 中的 batch_inputs 形状：
    #   [batch_size, history_steps, len(FEATURE_COLUMNS)]
    # loader 中的 batch_targets 形状：
    #   [batch_size, prediction_steps, len(TARGET_COLUMNS)]
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            preds = model(batch_inputs)
            loss = criterion(preds, batch_targets)
            total_loss += loss.item() * len(batch_inputs)
            total_count += len(batch_inputs)
    return total_loss / max(total_count, 1)


def train(args: argparse.Namespace) -> dict:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    # 训练数据在这里真正进入训练流程。
    # build_sequences 会：
    # 1. 扫描 dataset_dir 下的所有 CSV 文件
    # 2. 读取原始驾驶日志，并补出 yaw、dt 等派生字段
    # 3. 按 history_steps 切出历史窗口，作为模型输入
    # 4. 按 prediction_steps 生成未来监督标签
    # 5. 划分训练集/验证集，并用训练集统计量做标准化
    train_loader, val_loader, bundle = build_sequences(
        dataset_dir=args.dataset_dir,
        history_steps=args.history_steps,
        prediction_steps=args.prediction_steps,
        validation_ratio=args.validation_ratio,
        batch_size=args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    # 这里实例化的训练模型是 carla_ml.model.DrivingLSTM。
    # 模型结构：
    #   LSTM 编码器 -> 取最后一个时间步的隐藏状态 -> MLP 预测头
    # 模型输入：
    #   input_size = len(FEATURE_COLUMNS) = 15
    #   单个样本形状为 [history_steps, 15]
    # 模型输出：
    #   target_size = len(TARGET_COLUMNS) = 5
    #   单个样本输出形状为 [prediction_steps, 5]
    #   5 个输出字段依次是：
    #   future_throttle, future_steering, future_dx_local, future_dy_local, future_dyaw
    model = DrivingLSTM(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        prediction_steps=args.prediction_steps,
        target_size=len(TARGET_COLUMNS),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = WeightedTrajectoryLoss(
        torch.tensor(bundle.target_mean, dtype=torch.float32, device=device),
        torch.tensor(bundle.target_std, dtype=torch.float32, device=device),
    )

    best_val = float("inf")
    patience = 0
    history = {"train_loss": [], "val_loss": []}
    best_path = MODELS_DIR / "driving_lstm.pt"
    metadata_path = MODELS_DIR / "driving_lstm_metadata.json"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_count = 0
        for batch_inputs, batch_targets in train_loader:
            # 每个 batch 都是在这里送入模型的。
            # batch_inputs：过去 history_steps 帧的车辆状态和控制量
            # batch_targets：以这个历史窗口最后一帧为基准，对未来 prediction_steps 帧构造的标签
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch_inputs)
            # preds 和 batch_targets 形状一致：
            # [batch_size, prediction_steps, len(TARGET_COLUMNS)]
            loss = criterion(preds, batch_targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_total += loss.item() * len(batch_inputs)
            train_count += len(batch_inputs)

        train_loss = train_loss_total / max(train_count, 1)
        val_loss = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            # 保存当前最优模型权重，以及重建模型结构所需的超参数。
            # run_carla_controller.py 启动时会加载这个文件，恢复同一个 DrivingLSTM。
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": len(FEATURE_COLUMNS),
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "prediction_steps": args.prediction_steps,
                    "target_size": len(TARGET_COLUMNS),
                    "dropout": args.dropout,
                    "history_steps": args.history_steps,
                },
                best_path,
            )
            # 同时保存训练集统计量。
            # 推理阶段会用这些统计量对实时历史窗口做标准化，再把模型输出反标准化回真实量纲。
            payload = {
                "feature_columns": FEATURE_COLUMNS,
                "target_columns": TARGET_COLUMNS,
                "feature_mean": bundle.feature_mean.tolist(),
                "feature_std": bundle.feature_std.tolist(),
                "target_mean": bundle.target_mean.tolist(),
                "target_std": bundle.target_std.tolist(),
                "history_steps": args.history_steps,
                "prediction_steps": args.prediction_steps,
                "best_val_loss": best_val,
            }
            metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            patience += 1
            if patience >= args.early_stopping:
                print(f"Early stopping at epoch {epoch}.")
                break

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="validation")
    ax.set_title("LSTM Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = PLOTS_DIR / "training_loss.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    return {
        "model_path": str(best_path),
        "metadata_path": str(metadata_path),
        "plot_path": str(plot_path),
        "best_val_loss": best_val,
        "device": str(device),
        "samples": len(bundle.inputs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM policy from CSV driving data.")
    # history_steps：模型每次输入时会看到多少帧历史数据
    # prediction_steps：模型一次会预测未来多少步
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--history-steps", type=int, default=20)
    parser.add_argument("--prediction-steps", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--early-stopping", type=int, default=5)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = train(parse_args())
    print(json.dumps(result, indent=2))
