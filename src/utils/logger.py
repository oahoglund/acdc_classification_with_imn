from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import re

def find_ckpts(info_path: Path) -> list:
    """Given a results info.json (recording fold-0's run_dir), return all 5 fold checkpoints in order."""
    import json
    info_path = Path(info_path)
    with open(info_path) as f:
        fold0_run_dir = Path(json.load(f)["local_run_path"])

    run_num = int("".join(filter(str.isdigit, fold0_run_dir.name)))
    log_root = fold0_run_dir.parent

    ckpts = []
    for i in range(5):
        run_dir = log_root / f"run{run_num + i}"
        best_ckpts = sorted(run_dir.glob("best-*.ckpt"))
        if not best_ckpts:
            raise RuntimeError(f"No best-*.ckpt found in {run_dir}")
        ckpts.append(str(best_ckpts[-1]))
    return ckpts


def get_next_run_dir(save_dir: str | Path) -> Path:
        base = Path(save_dir)
        base.mkdir(parents=True, exist_ok=True)

        run_pattern = re.compile(r"^run(\d+)$")

        runs = 0
        for p in base.iterdir():
            if p.is_dir():
                m = run_pattern.match(p.name)
                if m:
                    runs += 1

        next_run = (runs + 1) if runs else 1
        return base / f"run{next_run}"

class LocalImageLogger(Logger):
    def __init__(self, save_dir: str):
        super().__init__()
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)

        self.palette = np.array([
            [0,   0,   0],      # black
            [0, 255,   0],      # green
            [255, 0,   0],      # red
            [0,   0, 255],      # blue
            [255, 165, 0],      # orange
        ], dtype=np.uint8)

    @property
    def name(self):
        return "local_image_logger"

    @property
    def version(self):
        return "0"
    
    @property
    def save_dir(self):
        return str(self._save_dir)
    
    @save_dir.setter
    def save_dir(self, value):
        # Lightning may set this after init
        self._save_dir = Path(value)
        self._save_dir.mkdir(parents=True, exist_ok=True)
    
    def colorize(self, mask: np.ndarray) -> np.ndarray:
        return self.palette[mask]  # [H,W,3]

    def overlay(self, img_gray: np.ndarray, color: np.ndarray, alpha=0.45):
        # img_gray: [H,W] uint8
        img_rgb = np.stack([img_gray]*3, axis=-1)
        out = img_rgb * (1 - alpha) + color * alpha
        return out.astype(np.uint8)

    @rank_zero_only
    def log_image(self, name, img_gray, gt=None, pred=None):
        """
        img_gray: [H,W] uint8
        gt/pred:  [H,W] uint8 (class ids)
        """
        base = self._save_dir / name
        base.mkdir(exist_ok=True)
        print(base)

        # save raw grayscale
        imageio.imwrite(base / "image.png", img_gray)

        if gt is not None:
            imageio.imwrite(base / "gt.png", gt)
            gt_color = self.colorize(gt)
            imageio.imwrite(base / "overlay_gt.png",
                            self.overlay(img_gray, gt_color))

        if pred is not None:
            imageio.imwrite(base / "pred.png", pred)
            pr_color = self.colorize(pred)
            imageio.imwrite(base / "overlay_pred.png",
                            self.overlay(img_gray, pr_color))

    # required by Lightning
    def log_metrics(self, metrics, step):
        pass

    def log_hyperparams(self, params):
        pass



