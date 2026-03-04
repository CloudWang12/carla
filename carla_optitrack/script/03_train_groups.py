# script/03_train_groups.py
import os, sys, json, glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_bc import MLPPolicy

# -------------------------
# 配置区：你可以只改这里
# -------------------------
WINDOW = 5          # 滑窗长度（20Hz时≈0.25s）
EPOCHS = 25
BATCH = 512
LR = 1e-3

# 特征基础：speed + IMU（稳定，适合你当前数据的“混乱rot”情况）
BASE_FEATURES = [
    "linear_speed",
    "gyro_x", "gyro_y", "gyro_z",
    "acc_x", "acc_y", "acc_z",
]

LABEL_COLS = ["steering", "throttle"]

ROT_COLS = ["rot_0", "rot_1", "rot_2", "rot_3"]  # 原始列名
# 如果某个 group 内 rot 约定稳定，会自动启用 rot 并统一成 xyzw
# 否则自动降级为不使用 rot
AUTO_USE_ROT = True

# -------------------------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def find_dataset_root():
    """
    允许两种摆放：
    1) data/raw_csv/QCarDataSet/<group>/*.csv
    2) data/raw_csv/<group>/*.csv
    """
    p1 = os.path.join(PROJECT_ROOT, "data", "raw_csv", "QCarDataSet")
    p2 = os.path.join(PROJECT_ROOT, "data", "raw_csv")
    if os.path.isdir(p1):
        return p1
    return p2

def list_groups(dataset_root: str):
    groups = []
    for name in os.listdir(dataset_root):
        d = os.path.join(dataset_root, name)
        if os.path.isdir(d):
            # 有 csv 才算 group
            if glob.glob(os.path.join(d, "**", "*.csv"), recursive=True):
                groups.append(name)
    groups.sort()
    return groups

def infer_w_like_col(df_rot: pd.DataFrame):
    """
    用 abs 中位数判断哪个 rot_* 更像 w（越接近 1 越像 w）
    """
    m = df_rot.abs().median()
    return m.idxmax()

def normalize_fit(x: np.ndarray):
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + 1e-6
    return mu, sd

def normalize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (x - mu) / sd

def make_windows_by_file(df: pd.DataFrame, feature_cols: list[str], label_cols: list[str], window: int):
    Xs, Ys = [], []
    for _, g in df.groupby("__source_file__", sort=False):
        g = g.dropna(subset=feature_cols + label_cols)
        if len(g) < window:
            continue
        feat = g[feature_cols].to_numpy(np.float32)
        lab = g[label_cols].to_numpy(np.float32)
        for i in range(window - 1, len(g)):
            Xs.append(feat[i - window + 1:i + 1].reshape(-1))
            Ys.append(lab[i])
    if not Xs:
        raise RuntimeError("No windows built (data too short or missing columns).")
    return np.stack(Xs), np.stack(Ys)

def load_group_csvs(dataset_root: str, group: str) -> pd.DataFrame:
    gdir = os.path.join(dataset_root, group)
    paths = sorted(glob.glob(os.path.join(gdir, "**", "*.csv"), recursive=True))
    if not paths:
        raise RuntimeError(f"No csv under group: {group}")

    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["__source_file__"] = p
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    # 有 time 就按 time 排（更合理）
    if "time" in out.columns:
        out.sort_values(["__source_file__", "time"], inplace=True)
        out.reset_index(drop=True, inplace=True)
    return out

def maybe_enable_rot(df: pd.DataFrame):
    """
    如果 rot 列存在且约定基本一致，则统一成 xyzw（rot_0..rot_3 = x,y,z,w）加入特征。
    如果约定混乱（出现 rot_1/rot_2 为 w_like 等），则禁用 rot。
    """
    for c in ROT_COLS:
        if c not in df.columns:
            return False, None, df

    # 按文件判断 w_like_col
    w_like = []
    for _, g in df.groupby("__source_file__", sort=False):
        r = g[ROT_COLS].dropna()
        if len(r) < 10:
            continue
        w_like.append(infer_w_like_col(r))
    if not w_like:
        return False, None, df

    # 统计
    vc = pd.Series(w_like).value_counts()
    top = vc.index[0]
    ratio = vc.iloc[0] / vc.sum()

    # 只接受 rot_0 或 rot_3 作为 w（两种常见顺序）
    if top not in ["rot_0", "rot_3"] or ratio < 0.85:
        # 约定太混乱，禁用 rot
        return False, None, df

    # 将所有文件的 rot 统一为 xyzw
    # 如果 w 在 rot_3：认为原数据就是 xyzw
    # 如果 w 在 rot_0：认为原数据是 wxyz -> 转成 xyzw
    df2 = df.copy()
    if top == "rot_0":
        # wxyz -> xyzw
        # rot_0=w, rot_1=x, rot_2=y, rot_3=z  => x=rot_1,y=rot_2,z=rot_3,w=rot_0
        df2["rot_0"], df2["rot_1"], df2["rot_2"], df2["rot_3"] = (
            df2["rot_1"], df2["rot_2"], df2["rot_3"], df2["rot_0"]
        )
        quat_order = "xyzw_from_wxyz"
    else:
        quat_order = "xyzw"

    return True, quat_order, df2

def train_one_group(dataset_root: str, group: str):
    df = load_group_csvs(dataset_root, group)

    # 必需列检查
    need_basic = BASE_FEATURES + LABEL_COLS + ["__source_file__"]
    missing = [c for c in need_basic if c not in df.columns]
    if missing:
        raise RuntimeError(f"[{group}] missing columns: {missing}")

    use_rot = False
    quat_order = None
    if AUTO_USE_ROT:
        use_rot, quat_order, df = maybe_enable_rot(df)

    feature_cols = list(BASE_FEATURES)
    if use_rot:
        feature_cols += ROT_COLS  # 现在已经统一为 xyzw
    label_cols = LABEL_COLS

    X, Y = make_windows_by_file(df, feature_cols, label_cols, WINDOW)
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * 0.9)
    tr, va = idx[:split], idx[split:]

    x_mu, x_sd = normalize_fit(X[tr])
    y_mu, y_sd = normalize_fit(Y[tr])

    Xtr = normalize_apply(X[tr], x_mu, x_sd).astype(np.float32)
    Ytr = normalize_apply(Y[tr], y_mu, y_sd).astype(np.float32)
    Xva = normalize_apply(X[va], x_mu, x_sd).astype(np.float32)
    Yva = normalize_apply(Y[va], y_mu, y_sd).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
    val_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva))
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPPolicy(in_dim=X.shape[1], out_dim=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    for ep in range(EPOCHS):
        model.train()
        tr_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"[{group}] epoch {ep+1}/{EPOCHS}"):
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

        print(f"[{group}] train={tr_loss:.6f} val={va_loss:.6f}")

    out_dir = os.path.join(PROJECT_ROOT, "models", "groups", group)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "policy.pt"))

    meta = {
        "group": group,
        "window": WINDOW,
        "feature_cols": feature_cols,
        "label_cols": label_cols,
        "use_rot": bool(use_rot),
        "quat_order": quat_order,   # None/xyzw/xyzw_from_wxyz
        "x_mu": x_mu.tolist(),
        "x_sd": x_sd.tolist(),
        "y_mu": y_mu.tolist(),
        "y_sd": y_sd.tolist(),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved models for group={group} -> {out_dir}")

def main():
    dataset_root = find_dataset_root()
    groups = list_groups(dataset_root)
    if not groups:
        raise RuntimeError(f"No group folders found under: {dataset_root}")

    print("[INFO] dataset_root:", dataset_root)
    print("[INFO] groups:", len(groups))
    for g in groups:
        print(" -", g)

    for g in groups:
        train_one_group(dataset_root, g)

    print("[DONE] all groups trained.")

if __name__ == "__main__":
    main()