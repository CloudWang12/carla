import os
from src.dataset_bc import merge_all_csv

def main():
    # 以“当前脚本所在位置”推算项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_root = os.path.join(project_root, "data", "raw_csv")
    out_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    # 先打印一下你实际在找哪个目录，方便确认
    print("[INFO] project_root =", project_root)
    print("[INFO] raw_root     =", raw_root)

    df = merge_all_csv(raw_root)
    out_path = os.path.join(out_dir, "merged_all.csv")
    df.to_csv(out_path, index=False)

    print("[OK] saved:", out_path)
    print("[INFO] shape:", df.shape)

if __name__ == "__main__":
    main()