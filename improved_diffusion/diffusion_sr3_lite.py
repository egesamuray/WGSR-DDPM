# improved_diffusion/diffusion_sr3_lite.py
import math
import torch
import torch.nn.functional as F

def make_beta_schedule(T=1000, schedule="cosine", max_beta=0.999):
    if schedule == "linear":
        betas = torch.linspace(1e-4, 2e-2, T)
    elif schedule == "cosine":
        # Nichol & Dhariwal cosine schedule
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64)
        alphas_cumprod = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(max=max_beta).float()
    else:
        raise ValueError(f"Unknown schedule {schedule}")
    return betas.float()

class DiffusionConfig:
    def __init__(self, T=1000, schedule="cosine", device="cuda"):
        self.device = torch.device(device)
        self.T = T
        self.betas = make_beta_schedule(T=T, schedule=schedule).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]], dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

def extract(coeff, t, shape):
    # coeff: (T,), t: (B,)
    out = coeff.gather(-1, t).float()
    while out.dim() < len(shape):
        out = out.unsqueeze(-1)
    return out.expand(shape)

def q_sample(config: DiffusionConfig, x0, t, noise=None):
    """
    q(x_t | x_0) = N( sqrt(ᾱ_t) x_0, (1-ᾱ_t) I )
    """
    if noise is None:
        noise = torch.randn_like(x0)
    return extract(config.sqrt_alphas_cumprod, t, x0.shape) * x0 + \
           extract(config.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise

def predict_x0_from_eps(config: DiffusionConfig, x_t, t, eps_pred):
    return extract(config.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - \
           extract(config.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps_pred

@torch.no_grad()
def p_sample_step(config: DiffusionConfig, model, x_t, t, conditioning):
    eps_pred = model(x_t, t, conditioning=conditioning)
    x0_hat = predict_x0_from_eps(config, x_t, t, eps_pred)
    mean = extract(config.posterior_mean_coef1, t, x_t.shape) * x0_hat + \
           extract(config.posterior_mean_coef2, t, x_t.shape) * x_t
    log_var = extract(config.posterior_log_variance_clipped, t, x_t.shape)
    if (t == 0).all():
        return x0_hat
    noise = torch.randn_like(x_t)
    return mean + torch.exp(0.5 * log_var) * noise

@torch.no_grad()
def p_sample_loop(config: DiffusionConfig, model, shape, conditioning):
    B = shape[0]
    x_t = torch.randn(shape, device=config.device)
    for i in reversed(range(config.T)):
        t = torch.full((B,), i, device=config.device, dtype=torch.long)
        x_t = p_sample_step(config, model, x_t, t, conditioning)
    return x_t  # this is x0 (HF whitened) at the end
