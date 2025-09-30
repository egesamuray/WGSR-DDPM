# scripts/wavelet_compute_stats.py
import os, glob, numpy as np, torch
from PIL import Image
from tqdm import tqdm
import pywt

def to_tensor_rgb(im, size):
    if size is not None:
        im = im.resize((size, size), Image.BICUBIC)
    ar = torch.from_numpy(np.asarray(im.convert("RGB"))).float() / 127.5 - 1.0
    return ar.permute(2,0,1)  # C,H,W

def dwt_level1_rgb(x, wavelet="haar"):
    C,H,W = x.shape
    LLs, HFs = [], []
    for c in range(3):
        (LL,(LH,HL,HH)) = pywt.dwt2(x[c].cpu().numpy(), wavelet=wavelet, mode="periodization")
        LLs.append(torch.from_numpy(LL).float())
        HFs.extend([torch.from_numpy(LH).float(),
                    torch.from_numpy(HL).float(),
                    torch.from_numpy(HH).float()])
    return torch.stack(LLs,0), torch.stack(HFs,0)

def main(data_dir, large_size=256, wavelet="haar", out_path="wavelet_stats_j1.npz"):
    exts = (".png",".jpg",".jpeg",".bmp",".webp")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(data_dir, "**", f"*{e}"), recursive=True)
    if not paths: raise SystemExit(f"No images under {data_dir}")

    sum_ll = torch.zeros(3); sumsq_ll = torch.zeros(3)
    sum_hf = torch.zeros(9); sumsq_hf = torch.zeros(9)
    count = 0
    for p in tqdm(paths, desc="Stats"):
        hr = to_tensor_rgb(Image.open(p).convert("RGB"), large_size)
        LL, HF = dwt_level1_rgb(hr, wavelet=wavelet)
        sum_ll += LL.view(3,-1).mean(dim=1)
        sumsq_ll += LL.view(3,-1).pow(2).mean(dim=1)
        sum_hf += HF.view(9,-1).mean(dim=1)
        sumsq_hf += HF.view(9,-1).pow(2).mean(dim=1)
        count += 1

    mean_ll = (sum_ll / count).numpy()
    mean_hf = (sum_hf / count).numpy()
    var_ll  = (sumsq_ll / count).numpy() - mean_ll**2
    var_hf  = (sumsq_hf / count).numpy() - mean_hf**2
    std_ll  = np.sqrt(np.maximum(var_ll, 1e-8))
    std_hf  = np.sqrt(np.maximum(var_hf, 1e-8))
    np.savez(out_path, mean_ll=mean_ll, std_ll=std_ll, mean_hf=mean_hf, std_hf=std_hf, large_size=large_size, wavelet=wavelet)
    print(f"Saved stats to {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--large_size", type=int, default=256)
    ap.add_argument("--wavelet", type=str, default="haar")
    ap.add_argument("--out_path", type=str, default="wavelet_stats_j1.npz")
    args = ap.parse_args()
    main(**vars(args))
