import os, sys, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm



# =========================
#03_train_bc_model training part
# =========================
# 1) 从 data/processed/merged_all.csv 读取合并后的数据（包含多个原始CSV拼起来的数据）
# 2) 以 __source_file__ 为分组（每个原始CSV一个分组），在每个分组内做“滑动窗口”(WINDOW=5)
#    - 每个训练样本 X = 最近 5 帧的特征拼接（flatten 成一维向量）
#    - 标签 y = 当前帧（窗口最后一帧）的 steering / throttle
# 3) 对 X / y 做标准化（减均值除方差）
# 4) 训练一个 MLP（多层感知机）回归模型：输入 X，输出 [steering, throttle]
# 5) 保存模型权重 bc_policy.pt + 标准化统计信息 bc_policy_meta.json
#
# 注意：本脚本“训练阶段”不会真的在CARLA里做任何坐标系转换，它只是把CSV里 pos/rot 当做数值特征输入模型。
# 真正“坐标系如何对应/转换”发生在你后续运行 CARLA 推理脚本时（比如 feature_runtime/build_feature_vector），
# 需要保证训练时使用的 pos_x/pos_y/pos_z、rot_0..rot_3 的含义与推理阶段计算出来的含义一致。
#





# --- 让脚本可以 import src ---
# PROJECT_ROOT：项目根目录（假设本脚本在 script/ 下，因此 .. 回到项目根）

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 将项目根目录加入 sys.path，这样就能 import src.xxx 这种项目内部模块
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_bc import MLPPolicy  # 从 src.model_bc 导入 MLPPolicy 模型类
                                    # MLPPolicy 是一个 PyTorch 模型（MLP，多层全连接网络），用于做回归：
                                    # 输入一个向量 -> 输出两个连续值（steering 和 throttle）
# =========================
# 训练数据的“特征列”与“标签列”
# =========================
# FEATURE_COLS：模型输入特征（X）
# 这里包含：
# - linear_speed：线速度（通常单位 m/s）
# - gyro_x/y/z：陀螺仪角速度（通常单位 rad/s，需与你在线推理一致）
# - acc_x/y/z：加速度（通常单位 m/s^2，需与你在线推理一致）
# - pos_x/y/z：位置（坐标系/单位需与你在线推理一致；这里纯当数值特征）
# - rot_0..rot_3：旋转四元数（注意顺序可能是 xyzw 或 wxyz，训练/推理必须一致）
#
# 重要：本脚本并不对这些坐标系做转换。它默认 CSV 已经把这些值按某一约定写好了；
# 后续你在 CARLA 推理时也要按同样约定构造这些特征。

FEATURE_COLS = [
    "linear_speed",
    "gyro_x","gyro_y","gyro_z",
    "acc_x","acc_y","acc_z",
    "pos_x","pos_y","pos_z",
    "rot_0","rot_1","rot_2","rot_3"
]


LABEL_COLS = ["steering", "throttle"]# LABEL_COLS：模型输出标签（y）
                                    # 模型学习预测：
                                    # - steering：转向（通常是 -1~1 的归一化转向输入）
                                    # - throttle：油门（通常是 0~1 的归一化油门输入）



WINDOW = 5  # 用最近5帧拼起来作为输入（更稳）# WINDOW：滑动窗口长度（序列长度）
                                        # 例如 WINDOW=5 表示每个训练样本会用最近5帧的特征
                                        # 如果你的采样频率是20Hz，那么5帧对应0.25秒的历史信息


def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def fit_norm(x: np.ndarray):
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + 1e-6
    return mu, sd

def apply_norm(x, mu, sd):
    return (x - mu) / sd
# apply_norm：按给定 mu/sd 对数据做标准化
# 标准化公式： (x - mu) / sd


# make_windows：将 DataFrame 转换为训练用的 (X, Y)
# 核心思想：对每个原始CSV文件（用 __source_file__ 分组）单独做窗口，
# 避免不同文件之间跨边界拼接产生“伪序列”。
def make_windows(df: pd.DataFrame):
    Xs, Ys = [], []
    for _, g in df.groupby("__source_file__", sort=False):
        g = g.dropna(subset=FEATURE_COLS + LABEL_COLS)
        if len(g) < WINDOW:
            continue

        feat = g[FEATURE_COLS].to_numpy(np.float32)
        lab = g[LABEL_COLS].to_numpy(np.float32)

        for i in range(WINDOW - 1, len(g)):
            x = feat[i - WINDOW + 1 : i + 1].reshape(-1)   # [WINDOW*F]
            y = lab[i]                                     # last label # 构造滑窗样本：
            Xs.append(x)
            Ys.append(y)
            # i 从 WINDOW-1 到 T-1
            # x = feat[i-WINDOW+1 : i+1] => shape [WINDOW, F]
            # 然后 reshape(-1) => shape [WINDOW*F] 作为最终输入向量
            # y = lab[i] => 当前时刻（窗口最后一帧）的控制标签

    if not Xs:
        raise RuntimeError("No training windows built. Check columns / data.")
    return np.stack(Xs), np.stack(Ys)

def main():
    merged_path = os.path.join(PROJECT_ROOT, "data", "processed", "merged_all.csv")
    if not os.path.exists(merged_path):
        raise RuntimeError("merged_all.csv not found, run 02 first")

    df = pd.read_csv(merged_path)

    # 快速校验列是否齐全
    need = FEATURE_COLS + LABEL_COLS + ["__source_file__"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"merged_all missing columns: {missing}")

    X, Y = make_windows(df)
    print("[INFO] windows:", X.shape, Y.shape)

    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * 0.9)
    tr, va = idx[:split], idx[split:]

    x_mu, x_sd = fit_norm(X[tr])
    y_mu, y_sd = fit_norm(Y[tr])

    Xtr = apply_norm(X[tr], x_mu, x_sd).astype(np.float32)
    Ytr = apply_norm(Y[tr], y_mu, y_sd).astype(np.float32)
    Xva = apply_norm(X[va], x_mu, x_sd).astype(np.float32)
    Yva = apply_norm(Y[va], y_mu, y_sd).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
    val_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva))
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPPolicy(in_dim=X.shape[1], out_dim=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    epochs = 25
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"epoch {ep+1}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_ds)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                va_loss += loss_fn(pred, yb).item() * xb.size(0)
        va_loss /= len(val_ds)

        print(f"[EPOCH {ep+1}] train={tr_loss:.6f} val={va_loss:.6f}")

    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    model_path = os.path.join(PROJECT_ROOT, "models", "bc_policy.pt")
    meta_path = os.path.join(PROJECT_ROOT, "models", "bc_policy_meta.json")

    torch.save(model.state_dict(), model_path)
    meta = {
        "feature_cols": FEATURE_COLS,
        "label_cols": LABEL_COLS,
        "window": WINDOW,
        "x_mu": x_mu.tolist(),
        "x_sd": x_sd.tolist(),
        "y_mu": y_mu.tolist(),
        "y_sd": y_sd.tolist(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[OK] saved:", model_path)
    print("[OK] saved:", meta_path)

if __name__ == "__main__":
    main()