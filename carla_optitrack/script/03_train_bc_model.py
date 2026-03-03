import os, sys, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# --- 让脚本可以 import src ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_bc import MLPPolicy

FEATURE_COLS = [
    "linear_speed",
    "gyro_x","gyro_y","gyro_z",
    "acc_x","acc_y","acc_z",
    "pos_x","pos_y","pos_z",
    "rot_0","rot_1","rot_2","rot_3"
]
LABEL_COLS = ["steering", "throttle"]
WINDOW = 5  # 用最近5帧拼起来作为输入（更稳）

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def fit_norm(x: np.ndarray):
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + 1e-6
    return mu, sd

def apply_norm(x, mu, sd):
    return (x - mu) / sd

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
            y = lab[i]                                     # last label
            Xs.append(x)
            Ys.append(y)

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