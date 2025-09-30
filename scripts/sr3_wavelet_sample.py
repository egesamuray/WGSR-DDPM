# scripts/sr3_wavelet_sample.py
import os, glob, numpy as np, torch
from PIL import Image
from tqdm import tqdm
import torchvision.utils as vutils

from improved_diffusion.cond_unet import CondUNetModel
from improved_diffusion.diffusion_sr3_lite import DiffusionConfig, p_sample_loop
from improved_diffusion.sr3_wavelet_datasets import _to_tensor_rgb, dwt_level1_rgb, idwt_level1_rgb

def _load_stats(stats_npz):
    z = np.load(stats_npz)
    mean_hf = torch.tensor(z["mean_hf"]).float()
    std_hf  = torch.tensor(z["std_hf"]).float().clamp_min(1e-6)
    return mean_hf, std_hf

@torch.no_grad()
def dewhiten_hf(HFw, mean_hf, std_hf):
    return HFw * std_hf.view(9,1,1) + mean_hf.view(9,1,1)

def save_image_rgb(x, path):
    x = x.clamp(-1,1)
    x = ((x + 1.0) * 127.5).round().byte().cpu()
    vutils.save_image(x, path, normalize=False)

def main(
    input_dir,                   # directory of LR images (or HR if you like — we'll down/up sample)
    ckpt_path,                   # trained checkpoint (ema)
    out_dir="results/sr3_wavelet_samples",
    stats_npz="wavelet_stats_j1.npz",
    large_size=256,
    scale=4,
    wavelet="haar",
    T=1000,
    schedule="cosine",
    num_samples_per_image=1
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    hf_channels, cond_channels = cfg.get("hf_channels", 9), cfg.get("cond_channels", 3)

    model = CondUNetModel(in_channels_hf=hf_channels, cond_channels=cond_channels,
                          out_channels=hf_channels,
                          model_channels=128, num_res_blocks=2, channel_mult=(1,2,2,2),
                          num_heads=4, num_head_channels=None, dropout=0.0,
                          attention_resolutions="16,8",
                          conv_resample=True, dims=2, use_checkpoint=False,
                          num_heads_upsample=-1, use_scale_shift_norm=True, resblock_updown=True)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(device)

    # Diffusion config for sampling
    dcfg = DiffusionConfig(T=T, schedule=schedule, device=device.type)

    mean_hf, std_hf = _load_stats(stats_npz)

    # Enumerate input images
    exts = (".png",".jpg",".jpeg",".bmp",".webp")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(input_dir, f"*{e}"))
    if not paths: raise SystemExit(f"No images in {input_dir}")

    for p in tqdm(paths, desc="Sampling"):
        # Prepare conditioning from input LR (or HR->LR->up)
        hr = _to_tensor_rgb(Image.open(p).convert("RGB"), large_size).to(device)  # normalize to [-1,1]
        small = large_size // scale
        lr    = torch.nn.functional.interpolate(hr.unsqueeze(0), size=(small, small), mode="bicubic", align_corners=False).squeeze(0)
        lr_up = torch.nn.functional.interpolate(lr.unsqueeze(0), size=(large_size, large_size), mode="bicubic", align_corners=False).squeeze(0)

        # LL from DWT of LR↑ (acts as coarse guide)
        LL_est, _ = dwt_level1_rgb(lr_up, wavelet=wavelet)
        LL_est = LL_est.to(device)

        # conditioning at HF resolution (h = large_size/2)
        cond = torch.nn.functional.interpolate(lr_up.unsqueeze(0), size=LL_est.shape[-2:], mode="bilinear", align_corners=False)

        for k in range(num_samples_per_image):
            HFw_hat = p_sample_loop(dcfg, model, (1, hf_channels, LL_est.shape[-2], LL_est.shape[-1]), conditioning=cond)
            HF_hat  = dewhiten_hf(HFw_hat.squeeze(0).cpu(), mean_hf, std_hf).to(device)
            rec = idwt_level1_rgb(LL_est, HF_hat, wavelet=wavelet)  # (3,H,W)
            base = os.path.splitext(os.path.basename(p))[0]
            save_image_rgb(rec, os.path.join(out_dir, f"{base}_sr3w_{k}.png"))

    print(f"Saved samples to {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--out_dir", default="results/sr3_wavelet_samples")
    ap.add_argument("--stats_npz", default="wavelet_stats_j1.npz")
    ap.add_argument("--large_size", type=int, default=256)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--wavelet", type=str, default="haar")
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--schedule", type=str, default="cosine")
    ap.add_argument("--num_samples_per_image", type=int, default=1)
    args = ap.parse_args()
    main(**vars(args))
