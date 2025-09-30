# scripts/run_sr3_wavelet.py
import os, subprocess, sys, shlex

PY = sys.executable

def run(cmd):
    print(f"\n$ {cmd}")
    ret = subprocess.call(shlex.split(cmd))
    if ret != 0:
        raise SystemExit(f"Command failed with code {ret}")

def main(
    data_dir,                # training images root (folder of images)
    run_dir="runs/sr3_wavelet_demo",
    large_size=256,
    scale=4,
    wavelet="haar",
    T=1000,
    schedule="cosine",
    batch_size=16,
    lr=2e-4,
    max_steps=200000,
    save_interval=10000,
    lambda_kl=0.05,
    kl_weights=None,         # e.g. "[1,1,1.5, 1,1,1.5, 1,1,1.5]"
    amp=True,
):
    os.makedirs(run_dir, exist_ok=True)
    stats_npz = os.path.join(run_dir, "wavelet_stats_j1.npz")
    out_dir   = os.path.join(run_dir, "train")
    samples   = os.path.join(run_dir, "samples")

    # 1) Stats
    run(f'{PY} scripts/wavelet_compute_stats.py --data_dir "{data_dir}" --large_size {large_size} --wavelet {wavelet} --out_path "{stats_npz}"')

    # 2) Train
    cmd = f'{PY} scripts/sr3_wavelet_train.py --data_dir "{data_dir}" --out_dir "{out_dir}" ' \
          f'--stats_npz "{stats_npz}" --large_size {large_size} --scale {scale} --wavelet {wavelet} ' \
          f'--batch_size {batch_size} --lr {lr} --max_steps {max_steps} --save_interval {save_interval} ' \
          f'--T {T} --schedule {schedule} --lambda_kl {lambda_kl} '
    if kl_weights is not None:
        cmd += f'--kl_weights "{kl_weights}" '
    if amp:
        cmd += f'--amp '
    run(cmd)

    # 3) Sample (using final ckpt)
    ckpt = os.path.join(out_dir, "model_final.pt")
    run(f'{PY} scripts/sr3_wavelet_sample.py --input_dir "{data_dir}" --ckpt_path "{ckpt}" '
        f'--out_dir "{samples}" --stats_npz "{stats_npz}" --large_size {large_size} --scale {scale} '
        f'--wavelet {wavelet} --T {T} --schedule {schedule}')

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--run_dir", default="runs/sr3_wavelet_demo")
    ap.add_argument("--large_size", type=int, default=256)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--wavelet", type=str, default="haar")
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--schedule", type=str, default="cosine")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_steps", type=int, default=200000)
    ap.add_argument("--save_interval", type=int, default=10000)
    ap.add_argument("--lambda_kl", type=float, default=0.05)
    ap.add_argument("--kl_weights", type=str, default=None)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()
    main(**vars(args))
