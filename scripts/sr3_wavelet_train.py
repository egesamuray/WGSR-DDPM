# scripts/sr3_wavelet_train.py
"""
Train SR3 in the wavelet domain (level-1 DWT):
- Diffuse HF (9ch) in whitened space
- Condition on LR↑ (3ch), resized to (H/2,W/2) to match HF spatial size
- Loss = MSE(ε, ε̂) + λ_KL * KL( N(μ̂_c,σ̂_c^2) || N(0,1) ) over HF channels of x̂0 (whitened)
"""
import os, json, time, math, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from improved_diffusion.cond_unet import CondUNetModel
from improved_diffusion.diffusion_sr3_lite import DiffusionConfig, q_sample, predict_x0_from_eps
from improved_diffusion.sr3_wavelet_datasets import load_data_sr3_wavelet
from improved_diffusion.losses_wavelet import wavelet_hf_kl_regularizer

def default_unet_kwargs(num_channels=128, num_res_blocks=2, channel_mult=(1,2,2,2),
                        num_heads=4, num_head_channels=-1, dropout=0.0,
                        attention_resolutions="16,8"):
    # Map attention resolutions string (H/2) to scales if needed by your UNet; kept for compatibility.
    return dict(
        model_channels=num_channels,
        num_res_blocks=num_res_blocks,
        channel_mult=channel_mult,
        num_heads=num_heads,
        num_head_channels=num_head_channels if num_head_channels>0 else None,
        dropout=dropout,
        attention_resolutions=attention_resolutions,
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
    )

def save_ckpt(path, model, opt, step, cfg_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": cfg_dict}, path)

def load_kl_weights(kl_weights):
    if kl_weights is None: return None
    w = torch.tensor(kl_weights, dtype=torch.float32)
    assert w.numel()==9, "kl_weights must have 9 values (LH,HL,HH for R,G,B)"
    return w

def main(
    data_dir,
    out_dir="results/sr3_wavelet",
    stats_npz="wavelet_stats_j1.npz",
    large_size=256,
    scale=4,
    wavelet="haar",
    batch_size=16,
    lr=2e-4,
    ema_decay=0.9999,
    max_steps=200000,
    log_interval=100,
    save_interval=10000,
    seed=0,
    # diffusion
    T=1000,
    schedule="cosine",
    # unet
    num_channels=128,
    num_res_blocks=2,
    channel_mult=(1,2,2,2),
    num_heads=4,
    num_head_channels=-1,
    dropout=0.0,
    attention_resolutions="16,8",
    # losses
    lambda_kl=0.05,
    kl_weights=None,      # e.g. [1.0,1.0,1.5, 1.0,1.0,1.5, 1.0,1.0,1.5] to emphasize HH
    amp=True,
    num_workers=4,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(locals(), f, indent=2, default=str)

    # Data loader yields HFw (B,9,h,w) and KW["conditioning"] (B,3,h,w)
    data = load_data_sr3_wavelet(
        data_dir=data_dir, batch_size=batch_size, large_size=large_size,
        scale=scale, wavelet=wavelet, stats_npz=stats_npz, num_workers=num_workers
    )

    # Model
    hf_channels, cond_channels = 9, 3
    unet_kwargs = default_unet_kwargs(
        num_channels=num_channels, num_res_blocks=num_res_blocks, channel_mult=channel_mult,
        num_heads=num_heads, num_head_channels=num_head_channels, dropout=dropout,
        attention_resolutions=attention_resolutions
    )
    model = CondUNetModel(in_channels_hf=hf_channels, cond_channels=cond_channels, out_channels=hf_channels, **unet_kwargs).to(device)

    # EMA
    ema = CondUNetModel(in_channels_hf=hf_channels, cond_channels=cond_channels, out_channels=hf_channels, **unet_kwargs).to(device)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters(): p.requires_grad_(False)

    def update_ema(target, source, decay):
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(decay).add_(sp.data, alpha=1-decay)

    # Diffusion config
    dcfg = DiffusionConfig(T=T, schedule=schedule, device=device.type)

    # Optim
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    scaler = GradScaler(enabled=amp)
    kl_w = load_kl_weights(kl_weights).to(device) if kl_weights is not None else None

    step = 0
    t0 = time.time()
    while step < max_steps:
        HFw, KW = next(data)
        HFw = HFw.to(device)                       # (B,9,h,w)
        cond = KW["conditioning"].to(device)       # (B,3,h,w)
        B = HFw.size(0)

        t = torch.randint(0, dcfg.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(HFw)
        x_t = q_sample(dcfg, HFw, t, noise=noise)

        opt.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            eps_pred = model(x_t, t, conditioning=cond)
            loss_mse = F.mse_loss(eps_pred, noise)
            x0_hat = predict_x0_from_eps(dcfg, x_t, t, eps_pred)  # whitened HF estimate
            loss_kl = wavelet_hf_kl_regularizer(x0_hat, weights=kl_w)
            loss = loss_mse + lambda_kl * loss_kl

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        update_ema(ema, model, ema_decay)

        if step % log_interval == 0:
            elapsed = time.time() - t0
            print(f"[{step:07d}] loss={loss.item():.4f} (mse={loss_mse.item():.4f}, kl={loss_kl.item():.4f})  {elapsed/60:.1f} min")

        if (step > 0) and (step % save_interval == 0):
            ckpt = os.path.join(out_dir, f"model_step{step}.pt")
            cfg_dict = dict(T=T, schedule=schedule, hf_channels=hf_channels, cond_channels=cond_channels, large_size=large_size, scale=scale, wavelet=wavelet)
            save_ckpt(ckpt, ema, opt, step, cfg_dict)
            print(f"Saved checkpoint: {ckpt}")

        step += 1

    # final save
    ckpt = os.path.join(out_dir, f"model_final.pt")
    cfg_dict = dict(T=T, schedule=schedule, hf_channels=hf_channels, cond_channels=cond_channels, large_size=large_size, scale=scale, wavelet=wavelet)
    save_ckpt(ckpt, ema, opt, step, cfg_dict)
    print(f"Saved final checkpoint: {ckpt}")

if __name__ == "__main__":
    import argparse, ast
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="results/sr3_wavelet")
    ap.add_argument("--stats_npz", default="wavelet_stats_j1.npz")
    ap.add_argument("--large_size", type=int, default=256)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--wavelet", type=str, default="haar")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema_decay", type=float, default=0.9999)
    ap.add_argument("--max_steps", type=int, default=200000)
    ap.add_argument("--log_interval", type=int, default=100)
    ap.add_argument("--save_interval", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--schedule", type=str, default="cosine")
    ap.add_argument("--num_channels", type=int, default=128)
    ap.add_argument("--num_res_blocks", type=int, default=2)
    ap.add_argument("--channel_mult", type=str, default="1,2,2,2")
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_head_channels", type=int, default=-1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--attention_resolutions", type=str, default="16,8")
    ap.add_argument("--lambda_kl", type=float, default=0.05)
    ap.add_argument("--kl_weights", type=str, default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    # decode types
    args.channel_mult = tuple(map(int, args.channel_mult.split(",")))
    args.kl_weights = ast.literal_eval(args.kl_weights) if args.kl_weights is not None else None

    main(**vars(args))
