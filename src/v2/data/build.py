"""DataLoader construction."""
import os

import torch
from torch.utils.data import DataLoader

from src.v2.data.dataset import YoloDataset
from src.v2.utils.general import seed_worker

# Windows spawns workers as full processes: each one re-imports torch + cv2 and costs
# roughly this much RSS. Overshooting kills the whole run with
# "DataLoader worker exited unexpectedly", so the worker count is capped by free RAM.
WORKER_RAM_GB = 1.4


def build_dataset(img_dir, label_dir, nc, imgsz, hyp, augment, cache_ram=False, prefix=""):
    return YoloDataset(img_dir=img_dir, label_dir=label_dir, nc=nc, imgsz=imgsz,
                       hyp=hyp, augment=augment, cache_ram=cache_ram, prefix=prefix)


def safe_workers(requested, verbose=True):
    """Clamp worker count to CPU count and to available RAM."""
    nw = min(requested, os.cpu_count() or 1)
    try:
        import psutil
        avail = psutil.virtual_memory().available / 1e9
        cap = max(0, int((avail * 0.5) / WORKER_RAM_GB))
        if cap < nw:
            if verbose:
                print(f"  dataloader: {avail:.1f} GB RAM free -> capping workers "
                      f"{nw} -> {cap} (close browsers to raise this)")
            nw = cap
    except Exception:
        pass
    return max(nw, 0)


def build_dataloader(dataset, batch_size, workers, shuffle=True, seed=0, drop_last=False):
    nw = safe_workers(workers)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=nw,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=nw > 0,
                      prefetch_factor=2 if nw > 0 else None,
                      collate_fn=YoloDataset.collate_fn,
                      worker_init_fn=seed_worker,
                      drop_last=drop_last,
                      generator=generator)
