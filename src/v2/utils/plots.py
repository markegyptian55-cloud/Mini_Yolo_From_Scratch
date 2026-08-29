"""Training-curve plots from results.csv."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import pandas as pd               # noqa: E402


def plot_results(csv_path, out_path=None):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    out_path = Path(out_path) if out_path else csv_path.with_name("results.png")

    panels = [
        (["box_loss", "cls_loss", "l1_loss"], "loss components"),
        (["loss_o2m", "loss_o2o"], "one-to-many vs one-to-one"),
        (["mAP50", "mAP50_95"], "accuracy"),
        (["precision", "recall"], "precision / recall"),
        (["lr"], "learning rate"),
        (["prog_alpha"], "ProgLoss alpha"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), tight_layout=True)
    for ax, (cols, title) in zip(axes.ravel(), panels):
        for c in cols:
            if c in df:
                ax.plot(df["epoch"], df[c], linewidth=1.6, label=c, marker=".",
                        markersize=3)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import sys
    print(plot_results(sys.argv[1] if len(sys.argv) > 1 else "runs/v2/exp/results.csv"))
