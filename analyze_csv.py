# -*- coding: utf-8 -*-
"""读取 CSV 文件，输出简单统计并把图表保存为 PNG。

用法:
    python analyze_csv.py 数据.csv                 # 统计打印到屏幕，图表存到 数据_charts/
    python analyze_csv.py 数据.csv --show          # 保存之外再弹出图表窗口
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Windows 下让中文和负号正常显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def read_csv_any(path: Path) -> pd.DataFrame:
    """依次尝试常见编码读取 CSV（utf-8-sig 可兼容带 BOM 的文件）。"""
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"无法解码 {path}，请确认文件编码（已尝试 utf-8 / gbk）")


def print_stats(df: pd.DataFrame) -> None:
    print(f"共 {df.shape[0]} 行 × {df.shape[1]} 列\n")
    print("各列类型与缺失情况:")
    info = pd.DataFrame(
        {
            "类型": df.dtypes.astype(str),
            "缺失数": df.isna().sum(),
            "唯一值数": df.nunique(),
        }
    )
    print(info.to_string())

    num_cols = df.select_dtypes("number").columns
    if len(num_cols):
        print("\n数值列统计:")
        print(df[num_cols].describe().round(2).to_string())

    # 取值不多的列视为类别列，打印分布
    cat_cols = [c for c in df.columns if c not in num_cols and df[c].nunique() <= 20]
    for c in cat_cols:
        print(f"\n“{c}” 取值分布:")
        print(df[c].value_counts().to_string())


def plot_numeric(df: pd.DataFrame, outdir: Path, show: bool) -> Path | None:
    """所有数值列各画一个直方图，拼成一幅图。"""
    num_cols = list(df.select_dtypes("number").columns)
    if not num_cols:
        return None
    ncol = min(len(num_cols), 3)
    nrow = (len(num_cols) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, num_cols):
        df[col].dropna().plot.hist(ax=ax, bins=20, edgecolor="white")
        ax.set_title(f"{col} 分布")
        ax.set_xlabel(col)
    for ax in axes[len(num_cols):]:  # 关掉多余的子图
        ax.set_visible(False)
    fig.suptitle("数值列直方图", fontsize=14)
    fig.tight_layout()
    out = outdir / "直方图.png"
    fig.savefig(out, dpi=150)
    if not show:
        plt.close(fig)
    return out


def plot_correlation(df: pd.DataFrame, outdir: Path, show: bool) -> Path | None:
    """数值列两两相关系数热力图（至少 2 个数值列才画）。"""
    num_cols = list(df.select_dtypes("number").columns)
    if len(num_cols) < 2:
        return None
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(1.2 * len(num_cols) + 3, 1.0 * len(num_cols) + 2))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)), num_cols, rotation=30, ha="right")
    ax.set_yticks(range(len(num_cols)), num_cols)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="相关系数")
    ax.set_title("数值列相关系数")
    fig.tight_layout()
    out = outdir / "相关系数.png"
    fig.savefig(out, dpi=150)
    if not show:
        plt.close(fig)
    return out


def plot_categorical(df: pd.DataFrame, outdir: Path, show: bool) -> Path | None:
    """类别列取值计数柱状图（取值不超过 20 个的列，画前 20 个取值）。"""
    num_cols = df.select_dtypes("number").columns
    cat_cols = [c for c in df.columns if c not in num_cols and df[c].nunique() <= 20]
    if not cat_cols:
        return None
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(4.5 * len(cat_cols), 3.5))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cat_cols):
        counts = df[col].value_counts().head(20)
        ax.bar(counts.index.astype(str), counts.values, color="#4c72b0")
        ax.set_title(f"{col} 计数")
        ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(counts.values):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = outdir / "类别计数.png"
    fig.savefig(out, dpi=150)
    if not show:
        plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV 简单统计 + 画图")
    parser.add_argument("csv", help="CSV 文件路径")
    parser.add_argument("--show", action="store_true", help="保存之外再弹出图表窗口")
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.is_file():
        raise SystemExit(f"找不到文件: {path}")

    df = read_csv_any(path)
    print_stats(df)

    outdir = path.with_name(f"{path.stem}_charts")
    outdir.mkdir(exist_ok=True)
    saved = [p for p in (plot_numeric(df, outdir, args.show),
                         plot_correlation(df, outdir, args.show),
                         plot_categorical(df, outdir, args.show)) if p]
    print("\n图表已保存:")
    for p in saved:
        print(f"  {p}")

    if args.show and saved:
        plt.show()


if __name__ == "__main__":
    main()
