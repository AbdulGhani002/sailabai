"""GPU memory test from the first-two-weeks checklist.

Runs a few real training steps (forward, backward, optimiser) for each model with mixed precision
and reports torch.cuda.max_memory_allocated(). On Windows, set the NVIDIA Control Panel option
"CUDA - Sysmem Fallback Policy" to "Prefer No Sysmem Fallback" first; otherwise the driver silently
spills into system RAM and this test under-reports (and training crawls).
"""

from __future__ import annotations

import time

import torch

from sailab.forecast.inputs import MAP_CHANNELS, N_TOKENS
from sailab.nn.unet import ResNetUNet, count_parameters
from sailab.nn.unet_clstm import UNetConvLSTM
from sailab.nn.unet_tt import UNetTT
from sailab.torchutils import autocast


def _measure(name: str, model: torch.nn.Module, make_batch, loss_fn, steps: int, dev: torch.device, log) -> dict:
    model = model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    for _ in range(steps):
        batch = make_batch()
        with autocast(dev, True):
            out = model(*batch[:-1])
        loss = loss_fn(out.float(), batch[-1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize(dev)
    peak = torch.cuda.max_memory_allocated(dev) / 2**30
    secs = (time.time() - t0) / steps
    log(f"{name:28s} {count_parameters(model) / 1e6:5.1f}M params  peak {peak:5.2f} GB  {secs * 1000:6.0f} ms/step")
    del model, opt
    return {"model": name, "peak_gb": round(peak, 2), "ms_per_step": round(secs * 1000)}


def run_memtest(chip: int = 256, batch_size: int = 8, steps: int = 5, log=print) -> list[dict]:
    if not torch.cuda.is_available():
        log("no CUDA GPU found; nothing to measure")
        return []
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    log(f"{props.name}, {props.total_memory / 2**30:.1f} GB; chip {chip} px, batch {batch_size}, bf16 autocast")
    results = []
    ce = torch.nn.CrossEntropyLoss()
    for enc in ("resnet34",):
        results.append(_measure(
            f"Model 1 U-Net {enc}", ResNetUNet(7, 3, encoder=enc),
            lambda: (torch.randn(batch_size, 7, chip, chip, device=dev),
                     torch.randint(0, 3, (batch_size, chip, chip), device=dev)),
            ce, steps, dev, log))
    bce = torch.nn.BCEWithLogitsLoss()
    n_feat = 9
    for enc in ("resnet18", "resnet34"):
        results.append(_measure(
            f"Model 2 UNet-TT {enc}", UNetTT(len(MAP_CHANNELS), n_feat, N_TOKENS, encoder=enc),
            lambda: (torch.randn(batch_size, len(MAP_CHANNELS), chip, chip, device=dev),
                     torch.randn(batch_size, N_TOKENS, n_feat, device=dev),
                     torch.zeros(batch_size, N_TOKENS, n_feat, device=dev),
                     torch.zeros(batch_size, N_TOKENS, device=dev),
                     torch.rand(batch_size, device=dev) * 7,
                     torch.rand(batch_size, chip, chip, device=dev).round()),
            bce, steps, dev, log))
    results.append(_measure(
        "Model 2 ConvLSTM resnet18", UNetConvLSTM(len(MAP_CHANNELS), n_feat, N_TOKENS, encoder="resnet18"),
        lambda: (torch.randn(batch_size, len(MAP_CHANNELS), chip, chip, device=dev),
                 torch.randn(batch_size, N_TOKENS, n_feat, device=dev),
                 torch.zeros(batch_size, N_TOKENS, n_feat, device=dev),
                 torch.zeros(batch_size, N_TOKENS, device=dev),
                 torch.rand(batch_size, device=dev) * 7,
                 torch.rand(batch_size, chip, chip, device=dev).round()),
        bce, steps, dev, log))
    try:
        from terratorch.registry import BACKBONE_REGISTRY  # noqa: F401

        log("terratorch found: measure TerraMind-small with configs/terramind_small.yaml (terratorch fit)")
    except ImportError:
        log("TerraMind-small: install the `terramind` extra to include it (or measure on Kaggle)")
    return results
