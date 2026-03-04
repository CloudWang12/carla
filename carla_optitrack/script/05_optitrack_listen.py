import time
import csv
from dataclasses import dataclass

# pip install natnet
from natnet import NatNetClient

@dataclass
class Config:
    server_address: str = "127.0.0.1"  # Motive 运行的机器IP
    client_address: str = "127.0.0.1"  # 你本机网卡IP（同机可127）
    use_multicast: bool = True
    rigid_body_id: int | None = None   # 先填 None：自动取第一个刚体

OUT_CSV = "data/processed/optitrack_stream.csv"

def main():
    cfg = Config()

    # 写CSV头
    f = open(OUT_CSV, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["timestamp", "rb_id", "x", "y", "z", "qx", "qy", "qz", "qw"])

    def on_frame(frame):
        # frame.rigid_bodies: dict[id] -> rigid body data
        if not frame.rigid_bodies:
            return

        # 取指定刚体或第一个
        if cfg.rigid_body_id is None:
            rb_id, rb = next(iter(frame.rigid_bodies.items()))
        else:
            rb = frame.rigid_bodies.get(cfg.rigid_body_id)
            if rb is None:
                return
            rb_id = cfg.rigid_body_id

        # 注意：字段名依库实现可能是 rb.position / rb.rotation
        pos = rb.position          # (x,y,z) meters
        quat = rb.rotation         # (qx,qy,qz,qw) or similar —— 先print验证

        ts = time.time()
        # 如果 quat 顺序不确定，先把 quat 原样写进去，后面再统一
        w.writerow([ts, rb_id, pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]])

        # 控制台打印（低频）
        if int(ts * 2) % 2 == 0:
            print(f"RB[{rb_id}] pos={pos} quat={quat}")

    client = NatNetClient(
        server_address=cfg.server_address,
        client_address=cfg.client_address,
        use_multicast=cfg.use_multicast,
    )
    client.set_callback(on_data_frame_received=on_frame)
    client.start()

    print("[INFO] Listening OptiTrack... Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()
        f.close()
        print("[OK] saved:", OUT_CSV)

if __name__ == "__main__":
    main()