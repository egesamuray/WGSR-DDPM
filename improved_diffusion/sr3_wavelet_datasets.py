# improved_diffusion/sr3_wavelet_datasets.py
import os, glob
from typing import Optional, Tuple, Dict, List
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pywt  # pip install PyWavelets

def _to_tensor_rgb(im: Image.Image, size: Optional[int]) -> torch.Tensor:
    if size is not None:
        im = im.resize((size, size), Image.BICUBIC)
    ar = torch.from_numpy(np.asarray(im.convert("RGB"))).float() / 127.5 - 1.0
    return ar.permute(2, 0, 1)  # C,H,W

def dwt_level1_rgb(x: torch.Tensor, wavelet: str = "haar") -> Tuple[torch.Tensor, torch.Tensor]:
    C, H, W = x.shape
    assert C == 3
    LLs, HFs = [], []
    for c in range(C):
        arr = x[c].cpu().numpy()
        (LL, (LH, HL, HH)) = pywt.dwt2(arr, wavelet=wavelet, mode="periodization")
        LLs.append(torch.from_numpy(LL).float())
        HFs.extend([torch.from_numpy(LH).float(),
                    torch.from_numpy(HL).float(),
                    torch.from_numpy(HH).float()])
    LL = torch.stack(LLs, dim=0)      # (3,h,w)
    HF = torch.stack(HFs, dim=0)      # (9,h,w) order: [LH_R,HL_R,HH_R, LH_G,HL_G,HH_G, LH_B,HL_B,HH_B]
    return LL, HF

def idwt_level1_rgb(LL: torch.Tensor, HF: torch.Tensor, wavelet: str = "haar") -> torch.Tensor:
    outs = []
    for c in range(3):
        LH, HL, HH = HF[3*c+0].cpu().numpy(), HF[3*c+1].cpu().numpy(), HF[3*c+2].cpu().numpy()
        rec = pywt.idwt2((LL[c].cpu().numpy(), (LH, HL, HH)), wavelet=wavelet, mode="periodization")
        outs.append(torch.from_numpy(rec).float())
    out = torch.stack(outs, dim=0)
    return out.clamp_(-1, 1)

def _load_stats_npz(stats_path: Optional[str]):
    if not stats_path:
        return None
    if not os.path.exists(stats_path):
        return None
    z = np.load(stats_path)
    mean_hf = torch.tensor(z["mean_hf"]).float()  # (9,)
    std_hf  = torch.tensor(z["std_hf"]).float().clamp_min(1e-6)
    # LL stats available if saved, not required here:
    mean_ll = torch.tensor(z["mean_ll"]).float() if "mean_ll" in z else None
    std_ll  = torch.tensor(z["std_ll"]).float().clamp_min(1e-6) if "std_ll" in z else None
    return dict(mean_hf=mean_hf, std_hf=std_hf, mean_ll=mean_ll, std_ll=std_ll)

class SR3WaveletDataset(Dataset):
    """
    Produces:
        HF_target_whitened: (9,h,w)  - training target (whitened HF)
        KW={"conditioning": LR_up_rgb_at_h}: (3,h,w)
    """
    def __init__(
        self,
        root: str,
        large_size: int = 256,
        scale: int = 4,
        wavelet: str = "haar",
        stats_npz: Optional[str] = None,
    ):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.large_size = large_size
        self.scale = scale
        self.wavelet = wavelet
        self.stats = _load_stats_npz(stats_npz)

        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        self.paths: List[str] = []
        for ext in exts:
            self.paths.extend(glob.glob(os.path.join(self.root, "**", f"*{ext}"), recursive=True))
        if not self.paths:
            raise FileNotFoundError(f"No images found under {self.root}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        hr = _to_tensor_rgb(Image.open(p).convert("RGB"), self.large_size)  # (3,H,W) in [-1,1]

        small = self.large_size // self.scale
        lr    = F.interpolate(hr.unsqueeze(0), size=(small, small), mode="bicubic", align_corners=False).squeeze(0)
        lr_up = F.interpolate(lr.unsqueeze(0), size=(self.large_size, self.large_size), mode="bicubic", align_corners=False).squeeze(0)

        LL, HF = dwt_level1_rgb(hr, wavelet=self.wavelet)  # (3,h,w), (9,h,w)
        if self.stats is not None:
            HFw = (HF - self.stats["mean_hf"].view(9,1,1)) / self.stats["std_hf"].view(9,1,1)
        else:
            HFw = (HF - HF.mean(dim=(1,2), keepdim=True)) / (HF.std(dim=(1,2), keepdim=True).clamp_min(1e-6))

        KW: Dict[str, torch.Tensor] = {
            "conditioning": F.interpolate(lr_up.unsqueeze(0), size=LL.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
        }
        return HFw, KW

def load_data_sr3_wavelet(
    *,
    data_dir: str,
    batch_size: int,
    large_size: int = 256,
    scale: int = 4,
    wavelet: str = "haar",
    stats_npz: Optional[str] = None,
    deterministic: bool = False,
    num_workers: int = 4,
):
    ds = SR3WaveletDataset(
        data_dir, large_size=large_size, scale=scale, wavelet=wavelet, stats_npz=stats_npz
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=not deterministic, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )
    while True:
        for X, KW in loader:
            yield X, KW
