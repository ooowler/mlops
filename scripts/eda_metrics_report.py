import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.table import Table

from fraud_detection.config import DATA_PROCESSED, TARGET_COLUMN
from fraud_detection.errors import DataNotFoundError

OUTDIR = ROOT / "data" / "meta" / "eda_report"


def main():
    train_path = DATA_PROCESSED / "train.csv"
    test_path = DATA_PROCESSED / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise DataNotFoundError("Нет train.csv/test.csv — сначала preprocess или dvc repro")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    console = Console()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    vc_tr = train[TARGET_COLUMN].value_counts().sort_index()
    vc_te = test[TARGET_COLUMN].value_counts().sort_index()
    n_tr, n_te = len(train), len(test)
    fraud_tr = int((train[TARGET_COLUMN] == 1).sum())
    fraud_te = int((test[TARGET_COLUMN] == 1).sum())

    table = Table(title="Дисбаланс классов (train / test)")
    table.add_column("Выборка", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("Class 0", justify="right")
    table.add_column("Class 1 (fraud)", justify="right")
    table.add_column("Доля fraud", justify="right", style="yellow")
    table.add_row(
        "train",
        str(n_tr),
        str(int(vc_tr.get(0, 0))),
        str(int(vc_tr.get(1, 0))),
        f"{fraud_tr / n_tr:.4%}",
    )
    table.add_row(
        "test",
        str(n_te),
        str(int(vc_te.get(0, 0))),
        str(int(vc_te.get(1, 0))),
        f"{fraud_te / n_te:.4%}",
    )
    console.print(table)

    both = pd.concat([train, test], ignore_index=True)
    feat_cols = [c for c in both.columns if c != TARGET_COLUMN]
    ranked = []
    for col in feat_cols:
        g0 = both.loc[both[TARGET_COLUMN] == 0, col]
        g1 = both.loc[both[TARGET_COLUMN] == 1, col]
        m0, m1 = float(g0.mean()), float(g1.mean())
        ranked.append((col, m0, m1, abs(m1 - m0)))
    ranked.sort(key=lambda x: -x[3])
    top_n = 10
    top = ranked[:top_n]

    diff_table = Table(title="Топ признаков по |среднее(fraud) − среднее(normal)| (train+test)")
    diff_table.add_column("Признак", style="cyan")
    diff_table.add_column("mean 0", justify="right")
    diff_table.add_column("mean 1", justify="right")
    diff_table.add_column("|diff|", justify="right", style="yellow")
    for col, m0, m1, ad in top:
        diff_table.add_row(col, f"{m0:.4f}", f"{m1:.4f}", f"{ad:.4f}")
    console.print(diff_table)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Normal (0)", "Fraud (1)"]
    counts = [int(vc_tr.get(0, 0)) + int(vc_te.get(0, 0)), int(vc_tr.get(1, 0)) + int(vc_te.get(1, 0))]
    ax.bar(labels, counts, color=["steelblue", "coral"])
    ax.set_ylabel("Количество")
    ax.set_title("Распределение классов (train+test)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "class_balance.png", dpi=120)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    a0 = both.loc[both[TARGET_COLUMN] == 0, "Amount"]
    a1 = both.loc[both[TARGET_COLUMN] == 1, "Amount"]
    ax2.boxplot([a0, a1], labels=["0 Normal", "1 Fraud"], patch_artist=True)
    for p, c in zip(ax2.artists, ["steelblue", "coral"]):
        p.set_facecolor(c)
        p.set_alpha(0.65)
    ax2.set_ylabel("Amount")
    ax2.set_title("Amount по классу")
    fig2.tight_layout()
    fig2.savefig(OUTDIR / "amount_by_class.png", dpi=120)
    plt.close(fig2)

    plot_cols = [r[0] for r in top[:4]]
    fig3, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, col in zip(axes.flat, plot_cols):
        b0 = both.loc[both[TARGET_COLUMN] == 0, col]
        b1 = both.loc[both[TARGET_COLUMN] == 1, col]
        ax.boxplot([b0, b1], labels=["0", "1"], patch_artist=True)
        for p, c in zip(ax.artists, ["steelblue", "coral"]):
            p.set_facecolor(c)
            p.set_alpha(0.65)
        ax.set_title(col)
        ax.set_ylabel("значение")
    fig3.suptitle("Топ-4 признака по разнице средних (0 vs 1)", fontsize=11)
    fig3.tight_layout()
    fig3.savefig(OUTDIR / "features_top_by_class.png", dpi=120)
    plt.close(fig3)

    console.print(f"[green]PNG:[/green] {OUTDIR / 'class_balance.png'}")
    console.print(f"[green]PNG:[/green] {OUTDIR / 'amount_by_class.png'}")
    console.print(f"[green]PNG:[/green] {OUTDIR / 'features_top_by_class.png'}")


if __name__ == "__main__":
    main()
