from pathlib import Path
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
from scipy.stats import chisquare
from scipy.optimize import brentq
from itertools import combinations

sns.set_theme(style="whitegrid")


# ═══════════════════════════════════════════════════════════════════════════
# Standalone information-theory helpers (used by several plots below)
# ═══════════════════════════════════════════════════════════════════════════

def binary_entropy(p):
    """
    H(p) = -p*log2(p) - (1-p)*log2(1-p): Shannon entropy, in bits, of a
    Bernoulli(p) source. Max value is 1 bit at p = 0.5. Defined as 0 at
    p = 0 and p = 1 (clipped to avoid log(0)).
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def kl_divergence_binary(p, q):
    """D(Bernoulli(p) || Bernoulli(q)), in bits."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return p * np.log2(p / q) + (1 - p) * np.log2((1 - p) / (1 - q))


def invert_binary_entropy(h):
    """
    Given h in [0, 1] (bits), returns p in [0, 0.5] such that H(p) = h.
    H(p) is symmetric around p = 0.5 and strictly increasing on [0, 0.5],
    so this branch is the unique, well-defined choice.
    """
    h = min(max(float(h), 0.0), 1.0)
    if h <= 1e-9:
        return 0.0
    if h >= 1 - 1e-9:
        return 0.5
    return brentq(lambda p: binary_entropy(p) - h, 1e-9, 0.5 - 1e-9)


def plot_bit_entropy_curve(df, output_dir, filename="bit_entropy_curve",
                            p_col="bernoulli_p", h_col="bit_entropy",
                            algorithm_col="algorithm"):
    """
    Standalone plotting function -- no PlotGenerator instance required.

    df : DataFrame (or anything pandas can build one from) with at least
         the columns named by p_col, h_col and algorithm_col. Each row is
         one measured (p, h) pair, as produced by
         bit_entropy.compute_bit_level_entropy() for one file/run.

    Draws:
      - the theoretical H(p) curve as a black background curve
      - a dashed vertical reference line at p = 0.5 (maximum uncertainty)
      - a scatterplot of the real measured (p, h) points, colored by
        algorithm

    Saves ONLY a high-resolution PNG -- no PDF, as requested.

    Returns the Path to the saved PNG.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Same overplotting risk as binary_entropy_curve(): with many rows,
    # one algorithm's block can completely bury the others since seaborn
    # draws hue groups one at a time. Sample for speed, jitter for
    # visibility.
    plot_df = sample_for_scatter(df, group_col=algorithm_col, max_per_group=3000)
    algos_present = sorted(plot_df[algorithm_col].dropna().unique()) if algorithm_col in plot_df.columns else []
    if algos_present:
        jitter_step = (plot_df[p_col].max() - plot_df[p_col].min() or 1.0) * 0.01
        jitter_map = {
            alg: (i - (len(algos_present) - 1) / 2) * jitter_step
            for i, alg in enumerate(algos_present)
        }
        plot_df[p_col] = plot_df[p_col] + plot_df[algorithm_col].map(jitter_map).fillna(0)

    p_grid = np.linspace(0.0, 1.0, 500)
    h_grid = binary_entropy(p_grid)

    plt.figure(figsize=(9, 7))
    plt.plot(p_grid, h_grid, color="black", linewidth=2, zorder=1,
              label="Theoretical H(p) curve")
    plt.axvline(0.5, color="#555555", linestyle="--", linewidth=1.3, zorder=1,
                label="p = 0.5 (maximum uncertainty)")

    sns.scatterplot(
        data=plot_df, x=p_col, y=h_col, hue=algorithm_col,
        alpha=0.7, s=45, edgecolor="white", linewidth=0.4, zorder=2
    )

    plt.xlabel("Bernoulli parameter p  (fraction of bits = 1)")
    plt.ylabel("Binary entropy H(p)  [bits]")
    plt.title("Bit-Level Binary Entropy vs Bernoulli Parameter")
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.05)
    plt.legend(title="Algorithm", frameon=True, loc="lower center")
    plt.tight_layout()

    out_path = output_dir / f"{filename}.png"
    plt.savefig(out_path, dpi=350)  # PNG only, no PDF
    plt.close()

    return out_path


def sample_for_scatter(df, group_col="algorithm", max_per_group=3000, seed=42):
    """
    Stratified random sample capped at `max_per_group` rows per value of
    `group_col`, then SHUFFLED across groups before returning.

    Two problems this fixes for dense scatter plots (hundreds of
    thousands of rows, heavily overlapping points):
      1. Speed: plotting every raw point is slow and adds nothing visual
         once points are this dense -- a few thousand per group is enough
         to show the true shape of the distribution.
      2. Overplotting masking: seaborn draws one hue group at a time, in
         a fixed block order (e.g. all of AES, then all of DES, then all
         of RC4). When groups land on nearly the same positions, whichever
         group is drawn LAST completely covers the earlier ones -- so a
         legend can correctly list three algorithms while the plot only
         visually shows the last one. Shuffling the sampled rows across
         groups interleaves the draw order, so every color gets a fair
         chance to show through.
    """
    if group_col not in df.columns:
        return df.sample(n=min(len(df), max_per_group), random_state=seed) if len(df) > max_per_group else df

    parts = [
        group.sample(n=min(len(group), max_per_group), random_state=seed)
        for _, group in df.groupby(group_col)
    ]
    sampled = pd.concat(parts, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


class PlotGenerator:

    def __init__(self, csv_file, output_dir):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_files = self._resolve_csv_files(csv_file)
        if not self.csv_files:
            raise FileNotFoundError(
                f"No experiment CSV matched: {csv_file}"
            )

        frames = []
        for path in self.csv_files:
            frame = pd.read_csv(path)
            frame["source_csv"] = path.stem
            frames.append(frame)
        self.df = pd.concat(frames, ignore_index=True)

        self.pdf = self.df.copy()
        # Extensions
        self.pdf["extension"] = (
            self.pdf["file_name"]
            .str.split(".")
            .str[-1]
            .str.lower()
        )

        self.algorithms = sorted(self.pdf["algorithm"].dropna().unique())
        self.multi_algorithm = len(self.algorithms) > 1

    @staticmethod
    def _resolve_csv_files(csv_file):
        """
        Accepts:
          - a single path/string pointing at one CSV (e.g. "data/csv/aes.csv")
          - a list/tuple of paths
          - a glob pattern (e.g. "data/csv/*.csv") to combine several
            per-algorithm files into one dataset for cross-algorithm plots.
        Always excludes '*_keys.csv' files: those hold cryptographic
        material, never experiment metrics, and must never end up plotted.
        """
        if isinstance(csv_file, (list, tuple)):
            candidates = [Path(p) for p in csv_file]
        else:
            csv_file = str(csv_file)
            if any(ch in csv_file for ch in "*?[]"):
                candidates = [Path(p) for p in sorted(glob.glob(csv_file))]
            else:
                candidates = [Path(csv_file)]

        return [
            p for p in candidates
            if p.exists() and not p.stem.endswith("_keys")
        ]

    # ═════════════════════════════════════════════════════════════════════
    # Original plots (unchanged in spirit; a few made hue/algorithm-aware
    # so they stay meaningful when several per-algorithm CSVs are combined)
    # ═════════════════════════════════════════════════════════════════════

    def entropy_histogram(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(
            self.pdf["entropy_raw"],
            color="#2c7bb6",
            kde=True,
            label="Original",
            alpha=0.6,
            stat="density"
        )
        sns.histplot(
            self.pdf["entropy_encrypted"],
            color="#d7191c",
            kde=True,
            label="Encrypted",
            alpha=0.6,
            stat="density"
        )
        plt.axvline(8.0, color='black', linestyle='--', linewidth=1.2,
                    label='Theoretical max (8 bits)')
        mean_enc = self.pdf["entropy_encrypted"].mean()
        plt.axvline(mean_enc, color='#d7191c', linestyle=':', linewidth=1.5,
                    label=f'Mean encrypted = {mean_enc:.3f}')

        plt.title("Entropy Distribution: Original vs Encrypted Data")
        plt.xlabel("Shannon Entropy (bits per byte)")
        plt.ylabel("Probability Density")
        plt.legend(frameon=True, loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir / "entropy_histogram.png", dpi=350)
        plt.close()

    def entropy_delta(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.pdf, x="entropy_delta", kde=True, bins=30)
        plt.title("Entropy Gain")
        plt.tight_layout()
        plt.savefig(self.output_dir / "entropy_delta.png", dpi=350)
        plt.close()

    def encryption_times(self):
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=self.pdf, x="algorithm", y="encrypt_time_ms")
        plt.title("Encryption Time Comparison")
        plt.tight_layout()
        plt.savefig(self.output_dir / "encryption_time.png", dpi=350)
        plt.close()

    def entropy_delta_boxplot(self):
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=self.pdf, x="algorithm", y="entropy_delta",
                    palette="colorblind")
        sns.stripplot(data=self.pdf, x="algorithm", y="entropy_delta",
                      color="black", alpha=.25, size=2)
        plt.ylabel("Entropy Gain (bits)")
        plt.xlabel("Algorithm")
        plt.title("Entropy Gain by Encryption Algorithm")
        plt.tight_layout()
        plt.savefig(self.output_dir / "entropy_delta_boxplot.png", dpi=350)
        plt.close()

    def decryption_times(self):
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=self.pdf, x="algorithm", y="decrypt_time_ms")
        plt.title("Decryption Time Comparison")
        plt.tight_layout()
        plt.savefig(self.output_dir / "decryption_time.png", dpi=350)
        plt.close()

    def entropy_scatter(self):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.pdf, x="entropy_raw", y="entropy_encrypted")
        plt.title("Original vs Encrypted Entropy")
        plt.tight_layout()
        plt.savefig(self.output_dir / "entropy_scatter.png", dpi=350)
        plt.close()

    def descriptive_statistics(self):
        stats = (
            self.df
            .groupby("algorithm")
            .agg(
                mean_raw=("entropy_raw", "mean"),
                mean_encrypted=("entropy_encrypted", "mean"),
                std_encrypted=("entropy_encrypted", "std"),
                mean_encrypt_ms=("encrypt_time_ms", "mean"),
                mean_decrypt_ms=("decrypt_time_ms", "mean")
            )
            .reset_index()
        )
        stats.to_csv(self.output_dir / "descriptive_statistics.csv", index=False)
        return stats

    def before_after_entropy(self):
        plt.figure(figsize=(7, 7))
        sns.scatterplot(data=self.pdf, x="entropy_raw", y="entropy_encrypted",
                         hue="algorithm", alpha=.55)
        plt.plot([0, 8], [0, 8], "--", color="black", linewidth=1,
                 label="No entropy change")
        plt.xlim(0, 8.05)
        plt.ylim(0, 8.05)
        plt.xlabel("Original Entropy")
        plt.ylabel("Encrypted Entropy")
        plt.title("Entropy Before and After Encryption")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "before_after_entropy.png", dpi=350)
        plt.close()

    def entropy_delta_extension(self):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.pdf, x="extension", y="entropy_delta",
                    palette="colorblind")
        plt.xticks(rotation=45)
        plt.ylabel("Entropy Gain")
        plt.tight_layout()
        plt.savefig(self.output_dir / "entropy_delta_extension.png", dpi=350)
        plt.close()

    def extension_statistics(self):
        stats = (
            self.pdf
            .groupby("extension")
            .agg(
                files=("extension", "count"),
                mean_raw=("entropy_raw", "mean"),
                mean_enc=("entropy_encrypted", "mean"),
                mean_delta=("entropy_delta", "mean"),
                std_delta=("entropy_delta", "std")
            )
            .reset_index()
        )
        stats.to_csv(self.output_dir / "extension_statistics.csv", index=False)
        return stats

    def entropy_delta_ecdf(self):
        plt.figure(figsize=(8, 6))
        sns.ecdfplot(data=self.pdf, x="entropy_delta")
        plt.xlabel("Entropy Gain")
        plt.ylabel("Cumulative Probability")
        plt.title("ECDF of Entropy Gain")
        plt.tight_layout()
        plt.savefig(self.output_dir / "entropy_delta_ecdf.png", dpi=350)
        plt.close()

    def entropy_summary(self):
        summary = pd.DataFrame({
            "Metric": [
                "Mean original entropy", "Mean encrypted entropy",
                "Mean entropy gain", "Median entropy gain",
                "Maximum entropy gain", "Minimum entropy gain"
            ],
            "Value": [
                self.pdf.entropy_raw.mean(), self.pdf.entropy_encrypted.mean(),
                self.pdf.entropy_delta.mean(), self.pdf.entropy_delta.median(),
                self.pdf.entropy_delta.max(), self.pdf.entropy_delta.min()
            ]
        })
        summary.to_csv(self.output_dir / "entropy_summary.csv", index=False)

    def wilcoxon_test(self):
        stat, pvalue = wilcoxon(self.pdf["entropy_raw"], self.pdf["entropy_encrypted"])
        with open(self.output_dir / "wilcoxon.txt", "w") as f:
            f.write(f"Statistic: {stat}\n")
            f.write(f"P-value: {pvalue}\n")
            if pvalue < 0.05:
                f.write("\nResult:\nReject H0.\nEntropy changed significantly.")
            else:
                f.write("\nResult:\nFail to reject H0.")
        return stat, pvalue

    def theoretical_vs_experimental(self):
        df = self.pdf.copy()
        df["theoretical_entropy"] = 8.0
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df.sort_values("run_id"), x="run_id", y="entropy_encrypted",
            hue="algorithm" if self.multi_algorithm else None,
            label=None if self.multi_algorithm else "Experimental"
        )
        plt.axhline(8.0, color="black", linestyle="--", label="Theoretical (Ideal Cipher)")
        plt.title("Theoretical vs Experimental Entropy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "theory_vs_experiment.png", dpi=350)
        plt.close()

    def entropy_error(self):
        df = self.pdf.copy()
        df["error"] = 8.0 - df["entropy_encrypted"]
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="algorithm", y="error")
        plt.title("Deviation from Theoretical Entropy (8 bits)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "entropy_error.png", dpi=350)
        plt.close()

    def entropy_delta_vs_time(self):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=self.pdf, x="encrypt_time_ms", y="entropy_delta",
                         hue="algorithm", alpha=.6)
        plt.xlabel("Encryption Time (ms)")
        plt.ylabel("Entropy Gain")
        plt.title("Entropy Gain versus Encryption Time")
        plt.tight_layout()
        plt.savefig(self.output_dir / "entropy_delta_vs_time.png", dpi=350)
        plt.close()

    def closeness_histogram(self):
        df = self.pdf.copy()
        df["closeness"] = df["entropy_encrypted"] / 8.0
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x="closeness", kde=True)
        plt.title("Closeness to Ideal Entropy (1.0 = perfect)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "closeness_hist.png", dpi=350)
        plt.close()

    def scatter_theory(self):
        df = self.pdf.copy()
        df["theoretical"] = 8.0
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x="entropy_encrypted", y="theoretical")
        plt.axvline(8.0, linestyle="--", color="black")
        plt.title("Experimental vs Theoretical Entropy")
        plt.tight_layout()
        plt.savefig(self.output_dir / "scatter_theory.png", dpi=350)
        plt.close()

    def mean_error_by_algorithm(self):
        df = self.pdf.copy()
        df["error"] = 8.0 - df["entropy_encrypted"]
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df, x="algorithm", y="error", estimator="mean", errorbar="sd")
        plt.title("Mean Deviation from Theoretical Entropy (8 bits)")
        plt.ylabel("Entropy Error (bits)")
        plt.xlabel("Algorithm")
        plt.tight_layout()
        plt.savefig(self.output_dir / "mean_error.png", dpi=350)
        plt.close()

    def entropy_boxplot(self):
        """Boxplot with automatic pairwise comparisons (only when >= 2 algorithms)."""
        plt.figure(figsize=(8, 6))
        sns.boxplot(
            data=self.pdf, x="algorithm", y="entropy_encrypted",
            palette="colorblind", legend=False
        )
        sns.stripplot(
            data=self.pdf, x="algorithm", y="entropy_encrypted",
            palette="colorblind", legend=False, color="black",
            size=3, alpha=0.3, dodge=False
        )

        pairs = list(combinations(self.algorithms, 2))
        if pairs:
            try:
                ax = plt.gca()
                annotator = Annotator(
                    ax, pairs, data=self.pdf, x="algorithm", y="entropy_encrypted"
                )
                annotator.configure(
                    test="Mann-Whitney", text_format="star", loc="inside",
                    comparisons_correction="bonferroni"
                )
                annotator.apply_and_annotate()
            except ImportError:
                print("statannotations not available. Skipping significance annotations.")
        else:
            print("[plots] entropy_boxplot: only one algorithm present, "
                  "skipping pairwise significance annotations.")

        plt.title("Encrypted Entropy by Algorithm with Pairwise Significance")
        plt.ylabel("Entropy (bits per byte)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "entropy_boxplot.png", dpi=350)
        plt.close()

    def entropy_ecdf(self):
        plt.figure(figsize=(8, 6))
        sns.ecdfplot(data=self.pdf, x="entropy_encrypted",
                     hue="algorithm" if self.multi_algorithm else None)
        plt.axvline(8.0, color='black', linestyle='--', linewidth=1.2,
                    label='Ideal cipher (8 bits)')
        plt.title("ECDF of Encrypted Entropy by Algorithm")
        plt.xlabel("Entropy (bits per byte)")
        plt.ylabel("Cumulative Probability")
        plt.legend(title="Algorithm")
        plt.tight_layout()
        plt.savefig(self.output_dir / "entropy_ecdf.png", dpi=350)
        plt.close()

    def entropy_violin_advanced(self):
        plt.figure(figsize=(8, 6))
        sns.violinplot(
            data=self.pdf, x="algorithm", y="entropy_encrypted",
            inner="box", palette="colorblind", linewidth=1.2
        )
        plt.title("Entropy Density by Algorithm (with quartiles)")
        plt.xlabel("Algorithm")
        plt.ylabel("Entropy (bits per byte)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "entropy_violin_adv.png", dpi=350)
        plt.close()

    def pairwise_entropy_heatmap(self):
        """Heatmap of Mann-Whitney p-values between algorithms. Needs >= 2 algorithms."""
        if not self.multi_algorithm:
            print("[plots] Skipping pairwise_entropy_heatmap: needs >= 2 "
                  "algorithms in the combined dataset.")
            return

        algorithms = self.algorithms
        n = len(algorithms)
        p_matrix = np.zeros((n, n))
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms):
                if i == j:
                    p_matrix[i, j] = 1.0
                else:
                    data1 = self.pdf[self.pdf["algorithm"] == alg1]["entropy_encrypted"]
                    data2 = self.pdf[self.pdf["algorithm"] == alg2]["entropy_encrypted"]
                    _, p = mannwhitneyu(data1, data2, alternative="two-sided")
                    p_matrix[i, j] = p
        mask = ~np.eye(n, dtype=bool)
        p_matrix[mask] = np.minimum(p_matrix[mask] * (n * (n - 1) / 2), 1.0)

        plt.figure(figsize=(7, 6))
        sns.heatmap(
            p_matrix, annot=True, fmt=".1e", xticklabels=algorithms,
            yticklabels=algorithms, cmap="mako_r", vmin=0, vmax=0.05,
            linewidths=0.5, cbar_kws={"label": "Corrected p-value"}
        )
        plt.title("Pairwise Comparison of Encrypted Entropy (Bonferroni corrected)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "pairwise_entropy_heatmap.png", dpi=350)
        plt.close()

    def time_vs_entropy_scatter(self):
        plt.figure(figsize=(8, 6))
        summary = self.pdf.groupby("algorithm").agg(
            mean_entropy=("entropy_encrypted", "mean"),
            mean_time=("encrypt_time_ms", "mean")
        ).reset_index()
        sns.scatterplot(data=summary, x="mean_time", y="mean_entropy",
                         style="algorithm", s=100)
        plt.axhline(8.0, color='gray', linestyle='--', label='Theoretical maximum')
        plt.title("Encryption Time vs Mean Entropy per Algorithm")
        plt.xlabel("Mean Encryption Time (ms)")
        plt.ylabel("Mean Encrypted Entropy (bits)")
        plt.legend(title="Algorithm")
        plt.tight_layout()
        plt.savefig(self.output_dir / "time_vs_entropy.png", dpi=350)
        plt.close()

    def combined_dashboard(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        sns.histplot(self.pdf["entropy_raw"], kde=True, color="#2c7bb6",
                     label="Original", ax=axes[0, 0], alpha=0.5)
        sns.histplot(self.pdf["entropy_encrypted"], kde=True, color="#d7191c",
                     label="Encrypted", ax=axes[0, 0], alpha=0.5)
        axes[0, 0].axvline(8, ls='--', color='black')
        axes[0, 0].set_title("Entropy Distribution")
        axes[0, 0].legend()

        sns.boxplot(data=self.pdf, x="algorithm", y="entropy_encrypted", ax=axes[0, 1])
        axes[0, 1].set_title("Encrypted Entropy by Algorithm")

        sns.ecdfplot(data=self.pdf, x="entropy_encrypted", ax=axes[1, 0])
        axes[1, 0].axvline(8, ls='--', color='black')
        axes[1, 0].set_title("Cumulative Distribution (ECDF)")

        df_err = self.pdf.copy()
        df_err["error"] = 8.0 - df_err["entropy_encrypted"]
        sns.barplot(data=df_err, x="algorithm", y="error", estimator="mean",
                    errorbar="sd", ax=axes[1, 1])
        axes[1, 1].set_title("Mean Deviation from 8 bits (error bars = SD)")
        axes[1, 1].set_ylabel("Error (bits)")

        plt.suptitle("Entropy Analysis Summary", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / "entropy_dashboard.png", dpi=350)
        plt.close()

    def wilcoxon_per_algorithm(self):
        results = []
        for alg in self.algorithms:
            subset = self.pdf[self.pdf["algorithm"] == alg]
            stat, p = wilcoxon(subset["entropy_raw"], subset["entropy_encrypted"])
            results.append({
                "algorithm": alg,
                "wilcoxon_stat": stat,
                "p_value": p,
                "median_delta": np.median(subset["entropy_delta"]),
                "mean_delta": subset["entropy_delta"].mean(),
                "std_delta": subset["entropy_delta"].std()
            })
        results_df = pd.DataFrame(results)
        results_df["p_corrected"] = (results_df["p_value"] * len(results_df)).clip(upper=1.0)
        results_df.to_csv(self.output_dir / "wilcoxon_by_algorithm.csv", index=False)
        with open(self.output_dir / "wilcoxon_report.txt", "w") as f:
            f.write("Wilcoxon Signed-Rank Test: Raw vs Encrypted Entropy per Algorithm\n\n")
            f.write(results_df.to_string(index=False))
            f.write("\n\nInterpretation: p_corrected < 0.05 indicates significant shift in entropy.\n")
        return results_df

    def kruskal_wallis_test(self):
        """Needs >= 2 algorithms present in the combined dataset."""
        if not self.multi_algorithm:
            print("[plots] Skipping kruskal_wallis_test: needs >= 2 algorithms.")
            return None, None

        groups = [group["entropy_encrypted"].values for _, group in self.pdf.groupby("algorithm")]
        H, p = kruskal(*groups)
        with open(self.output_dir / "kruskal_wallis.txt", "w") as f:
            f.write(f"Kruskal-Wallis H-statistic: {H:.4f}\n")
            f.write(f"P-value: {p:.6f}\n")
            if p < 0.05:
                f.write("Significant difference detected among algorithms (p<0.05).\n")
            else:
                f.write("No significant difference among algorithms.\n")
        return H, p

    # ═════════════════════════════════════════════════════════════════════
    # NEW: H(p) vs p per algorithm
    # ═════════════════════════════════════════════════════════════════════

    def binary_entropy_curve(self):
        """
        Theoretical H(p) curve with every experiment run overlaid.

        IMPORTANT: byte-level Shannon entropy over N bytes can never exceed
        log2(min(N, 256)) bits -- with fewer than 256 bytes you literally
        cannot have 256 distinct byte values, so the '8-bit ideal' is
        mathematically unreachable for small files regardless of cipher
        quality. Normalizing by a flat 8 made every small file look like a
        failing cipher even when it was performing at (or near) its true
        ceiling. Each run is now normalized by ITS OWN achievable ceiling:
        h_norm = entropy_encrypted / log2(min(size_bytes, 256)), clipped to
        1.0 for numerical safety. Both h_norm and H(p) live in [0,1], so
        h_norm is mapped to its equivalent Bernoulli parameter p_eq by
        numerically solving H(p_eq) = h_norm (branch p_eq <= 0.5). This is
        a visualization device, not a claim that ciphertext bytes behave
        like coin flips -- it puts every algorithm's runs on one universal
        reference curve so 'how close to maximal achievable uncertainty'
        is directly comparable, algorithm to algorithm, size-fairly.
        """
        p_grid = np.linspace(0.0, 1.0, 400)
        h_grid = binary_entropy(p_grid)

        # Sample BEFORE computing p_eq: brentq is a per-row numeric solve,
        # so sampling first avoids running it on hundreds of thousands of
        # rows.
        plot_df = sample_for_scatter(self.pdf, max_per_group=3000)
        max_achievable_bits = np.log2(np.clip(plot_df["size_bytes"], 2, 256))
        plot_df["h_norm"] = (plot_df["entropy_encrypted"] / max_achievable_bits).clip(upper=1.0)
        plot_df["p_eq"] = plot_df["h_norm"].apply(invert_binary_entropy)

        # Once size-adjusted, different algorithms land on nearly the SAME
        # (p, h) position (same files -> same ceiling), so seaborn's
        # per-group draw order (all of AES, then all of DES, then all of
        # RC4 -- fixed, NOT affected by row order) lets whichever group is
        # drawn last completely bury the others. A tiny per-algorithm
        # horizontal offset keeps every color visible without changing
        # the substance of the comparison.
        algos_present = sorted(plot_df["algorithm"].dropna().unique())
        jitter_step = 0.006
        jitter_map = {
            alg: (i - (len(algos_present) - 1) / 2) * jitter_step
            for i, alg in enumerate(algos_present)
        }
        plot_df["p_eq_jittered"] = plot_df["p_eq"] + plot_df["algorithm"].map(jitter_map).fillna(0)

        plt.figure(figsize=(9, 7))
        plt.plot(p_grid, h_grid, color="black", linewidth=1.5, label="H(p) theoretical curve")
        plt.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="p = 0.5 (max uncertainty)")
        sns.scatterplot(data=plot_df, x="p_eq_jittered", y="h_norm", hue="algorithm", alpha=0.6, s=35)

        plt.xlabel("Equivalent Bernoulli parameter p  (tiny per-algorithm offset for visibility)")
        plt.ylabel("Size-adjusted entropy  H(p) = entropy_encrypted / log2(min(size,256))")
        plt.title("H(p) vs p: each run vs its own achievable entropy ceiling")
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)
        plt.legend(title="Algorithm", loc="lower center")
        plt.tight_layout()
        plt.savefig(self.output_dir / "binary_entropy_curve.png", dpi=350)
        plt.close()

    def binary_entropy_curve_by_algorithm(self):
        """Same idea as binary_entropy_curve(), one panel per algorithm."""
        plot_df = sample_for_scatter(self.pdf, max_per_group=3000)
        max_achievable_bits = np.log2(np.clip(plot_df["size_bytes"], 2, 256))
        plot_df["h_norm"] = (plot_df["entropy_encrypted"] / max_achievable_bits).clip(upper=1.0)
        plot_df["p_eq"] = plot_df["h_norm"].apply(invert_binary_entropy)

        p_grid = np.linspace(0.0, 1.0, 400)
        h_grid = binary_entropy(p_grid)

        algorithms = self.algorithms if self.algorithms else ["data"]
        n = len(algorithms)
        fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5), sharey=True)
        if n == 1:
            axes = [axes]

        for ax, alg in zip(axes, algorithms):
            subset = plot_df[plot_df["algorithm"] == alg]
            ax.plot(p_grid, h_grid, color="black", linewidth=1.3)
            ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
            ax.scatter(subset["p_eq"], subset["h_norm"], alpha=0.5, s=25, color="#d7191c")
            ax.set_title(alg)
            ax.set_xlabel("p")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)

        axes[0].set_ylabel("H(p)  (size-adjusted entropy)")
        plt.suptitle("H(p) vs p by algorithm (size-adjusted)", y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / "binary_entropy_curve_by_algorithm.png", dpi=350)
        plt.close()

    def entropy_vs_size_ceiling(self):
        """
        The evidence plot: raw entropy_encrypted (not normalized) against
        file size on a log-x axis, with two theoretical references:

          1. The hard mathematical ceiling log2(min(size,256)) -- the
             absolute maximum possible for that sample size, unreachable
             except in the limit.
          2. The Miller-Madow bias-corrected EXPECTED value for an ideal
             (truly uniform) cipher output of that size:
                 E[H_hat] ~= 8 - (255)/(2*N*ln2) bits
             This is the plug-in entropy estimator's known downward bias
             for finite samples -- even a perfect cipher measures below
             8 bits for small N, by exactly this amount. Points hugging
             THIS curve (not the flat 8-bit line) are proof the cipher is
             behaving ideally, not proof of weakness.
        """
        df = self.pdf.copy()
        df = df[df["size_bytes"] > 0]
        df = sample_for_scatter(df, max_per_group=4000)

        size_grid = np.logspace(0, np.log10(max(df["size_bytes"].max(), 10)), 300)
        ceiling = np.log2(np.clip(size_grid, 2, 256))
        expected_ideal = np.clip(8.0 - 255.0 / (2.0 * size_grid * np.log(2)), 0, 8.0)

        plt.figure(figsize=(10, 7))
        plt.plot(size_grid, ceiling, color="black", linewidth=1.5, linestyle="--", zorder=1,
                  label="Hard ceiling  log2(min(size, 256))")
        plt.plot(size_grid, expected_ideal, color="#d7191c", linewidth=2.2, zorder=1,
                  label="Expected value for an IDEAL cipher (Miller-Madow correction)")
        plt.axhline(8.0, color="gray", linestyle=":", linewidth=1, zorder=1,
                    label="Flat 8-bit 'ideal' (only valid for size ≥ ~10,000 bytes)")

        sns.scatterplot(
            data=df, x="size_bytes", y="entropy_encrypted", hue="algorithm",
            alpha=0.35, s=18, edgecolor="none", zorder=2
        )

        plt.xscale("log")
        plt.xlabel("File size (bytes, log scale)")
        plt.ylabel("Encrypted entropy (bits/byte)")
        plt.title("Entropy vs File Size: why small files can't reach 8 bits")
        plt.legend(title="Algorithm", loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(self.output_dir / "entropy_vs_size_ceiling.png", dpi=350)
        plt.close()

    # ═════════════════════════════════════════════════════════════════════
    # NEW: distribution diagram
    # ═════════════════════════════════════════════════════════════════════

    def distribution_diagram(self):
        """
        Small-multiples of the distribution (KDE, or histogram+KDE if only
        one algorithm) for the main numeric variables, split by algorithm.
        One figure that answers 'what does each measured quantity look
        like, per algorithm' at a glance.

        Two things this corrects vs. a naive KDE:
          1. Time columns (encrypt_time_ms, decrypt_time_ms) span many
             orders of magnitude -- from a few ms to potentially minutes
             for large files -- so a LINEAR x-axis crushes almost every
             point against zero. Plotted on a log-x axis instead.
          2. When AES/DES/RC4 curves are nearly identical (as they are
             here), three overlapping SEMI-TRANSPARENT FILLS blend their
             colors into an indistinct grey/brown blob. Switched to
             outline-only curves (no fill) -- overlapping lines stay
             individually colored and readable even when they coincide.
        """
        variables = [
            ("entropy_raw", "Raw entropy (bits/byte)", False),
            ("entropy_encrypted", "Encrypted entropy (bits/byte)", False),
            ("encrypt_time_ms", "Encryption time (ms, log scale)", True),
            ("decrypt_time_ms", "Decryption time (ms, log scale)", True),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        axes = axes.flatten()

        for ax, (col, label, use_log) in zip(axes, variables):
            plot_df = self.pdf
            if use_log:
                # log-scale needs strictly positive values; drop non-positive
                # timings (near-instant operations rounding to 0ms) only for
                # this panel's KDE.
                plot_df = self.pdf[self.pdf[col] > 0]

            if self.multi_algorithm:
                sns.kdeplot(data=plot_df, x=col, hue="algorithm",
                            fill=False, linewidth=2, common_norm=False,
                            log_scale=use_log, ax=ax)
            else:
                sns.histplot(data=plot_df, x=col, kde=True, ax=ax,
                            color="#2c7bb6", log_scale=use_log)
            ax.set_xlabel(label)
            ax.set_ylabel("Density")

        plt.suptitle("Distribution Diagram: shape of every measured quantity", y=1.02, fontsize=15)
        plt.tight_layout()
        plt.savefig(self.output_dir / "distribution_diagram.png", dpi=350)
        plt.close()

    # ═════════════════════════════════════════════════════════════════════
    # NEW: convexity / concavity
    # ═════════════════════════════════════════════════════════════════════

    def entropy_concavity_demo(self):
        """
        Demonstrates that Shannon entropy H(p) is CONCAVE: for any two
        points p1, p2, the curve lies ABOVE the straight chord joining
        H(p1) and H(p2) (Jensen's inequality). The two anchor points come
        from the actual data (min/max mean normalized entropy observed
        across algorithms), so this is tied to the real experiment.
        """
        if self.multi_algorithm:
            means = self.pdf.groupby("algorithm")["entropy_encrypted"].mean() / 8.0
        else:
            means = pd.Series({"data": self.pdf["entropy_encrypted"].mean() / 8.0})

        p1, p2 = sorted(invert_binary_entropy(v) for v in (means.min(), means.max()))
        if p2 - p1 < 0.05:
            p1, p2 = max(0.05, p1 - 0.15), min(0.95, p2 + 0.15)

        p_grid = np.linspace(0.001, 0.999, 400)
        h_grid = binary_entropy(p_grid)
        h1, h2 = binary_entropy(p1), binary_entropy(p2)
        chord_p = np.array([p1, p2])
        chord_h = np.array([h1, h2])

        plt.figure(figsize=(8, 6))
        plt.plot(p_grid, h_grid, color="black", linewidth=2, label="H(p)")
        plt.plot(chord_p, chord_h, "o--", color="#d7191c", linewidth=1.5,
                 label="Chord between two observed points")
        plt.fill_between(
            p_grid, np.interp(p_grid, chord_p, chord_h), h_grid,
            where=(p_grid >= p1) & (p_grid <= p2),
            color="#2c7bb6", alpha=0.25,
            label="H(p) ≥ chord  ⇒  H is concave"
        )
        plt.xlabel("p")
        plt.ylabel("H(p) [bits]")
        plt.title("Concavity of Shannon Entropy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "entropy_concavity_demo.png", dpi=350)
        plt.close()

    def relative_entropy_convexity_demo(self, q=0.5):
        """
        Demonstrates that KL divergence D(p||q) is CONVEX in p (q fixed):
        the curve lies BELOW the chord joining any two points. q defaults
        to 0.5 -- the reference 'ideal cipher' distribution (fair coin,
        the maximum-entropy 2-outcome distribution).
        """
        p_grid = np.linspace(0.001, 0.999, 400)
        d_grid = kl_divergence_binary(p_grid, q)

        p1, p2 = 0.15, 0.85
        d1, d2 = kl_divergence_binary(p1, q), kl_divergence_binary(p2, q)

        plt.figure(figsize=(8, 6))
        plt.plot(p_grid, d_grid, color="black", linewidth=2, label=f"D(p || {q})")
        plt.plot([p1, p2], [d1, d2], "o--", color="#d7191c", linewidth=1.5,
                 label="Chord between two points")
        plt.fill_between(
            p_grid, d_grid, np.interp(p_grid, [p1, p2], [d1, d2]),
            where=(p_grid >= p1) & (p_grid <= p2),
            color="#2c7bb6", alpha=0.25,
            label="D(p||q) ≤ chord  ⇒  D is convex"
        )
        plt.xlabel("p")
        plt.ylabel(f"D(p || {q}) [bits]")
        plt.title("Convexity of Relative Entropy (KL Divergence)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "relative_entropy_convexity_demo.png", dpi=350)
        plt.close()

    # ═════════════════════════════════════════════════════════════════════
    # NEW: stochastic processes
    # ═════════════════════════════════════════════════════════════════════

    def entropy_stochastic_trajectory(self):
        """
        Treats the sequence of entropy_encrypted values (ordered by run_id
        within each algorithm) as one sample path of a stochastic process.
        A rolling mean/std band lets you visually check (weak) stationarity:
        if the band stays flat and of constant width as run_id grows, the
        process behaves as if it has constant mean and variance over
        'time' (loop index) -- what you'd expect from an ideal, memoryless
        cipher run repeatedly.
        """
        window = max(5, len(self.pdf) // 50)

        plt.figure(figsize=(11, 6))
        for alg in (self.algorithms or [None]):
            subset = (self.pdf[self.pdf["algorithm"] == alg] if alg is not None
                      else self.pdf).sort_values("run_id")
            if subset.empty:
                continue
            roll_mean = subset["entropy_encrypted"].rolling(window, min_periods=1).mean()
            roll_std = subset["entropy_encrypted"].rolling(window, min_periods=1).std().fillna(0)

            plt.plot(subset["run_id"], roll_mean, label=f"{alg} (rolling mean)")
            plt.fill_between(subset["run_id"], roll_mean - roll_std, roll_mean + roll_std, alpha=0.15)

        plt.axhline(8.0, color="black", linestyle="--", linewidth=1, label="Theoretical maximum (8 bits)")
        plt.xlabel("run_id (execution order)")
        plt.ylabel("Encrypted entropy (bits/byte)")
        plt.title("Entropy as a Stochastic Process: Sample Path & Rolling Stats")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "entropy_stochastic_trajectory.png", dpi=350)
        plt.close()

    def entropy_autocorrelation(self):
        """
        Autocorrelation of the entropy_encrypted sequence, per algorithm.
        If the cipher's output behaves as an i.i.d. random process run
        after run, the ACF should collapse to ~0 for every lag > 0 (white
        noise). A slowly decaying or oscillating ACF would suggest the
        key/IV generator -- or the cipher itself -- is leaking correlation
        between consecutive runs.
        """
        max_lag = 20

        plt.figure(figsize=(10, 6))
        for alg in (self.algorithms or [None]):
            subset = (self.pdf[self.pdf["algorithm"] == alg] if alg is not None
                      else self.pdf).sort_values("run_id")
            series = subset["entropy_encrypted"].to_numpy()
            n = len(series)
            if n < max_lag + 2:
                continue
            series = series - series.mean()
            denom = np.dot(series, series)
            lags = np.arange(0, max_lag + 1)
            acf = [
                np.dot(series[:n - lag], series[lag:]) / denom if denom > 0 else 0.0
                for lag in lags
            ]
            plt.plot(lags, acf, marker="o", markersize=3, label=alg)

        plt.axhline(0, color="black", linewidth=1)
        ci = 1.96 / np.sqrt(max(len(self.pdf), 1))
        plt.axhspan(-ci, ci, color="gray", alpha=0.2, label="~95% CI under white noise")
        plt.xlabel("Lag (in run order)")
        plt.ylabel("Autocorrelation")
        plt.title("Autocorrelation of Encrypted Entropy Across Runs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "entropy_autocorrelation.png", dpi=350)
        plt.close()

    def _acf_panel_grid(self, group_col, groups, filename, title, max_lag=15,
                        lag_unit="loops", ncols=4):
        """
        Shared helper: one ACF panel per value of `group_col` (a file name
        or an extension), one line per algorithm inside each panel,
        arranged in a grid (capped at `ncols` columns, wrapping into more
        rows as needed) instead of a single ever-widening row.

        Also fixes the earlier bug where the legend was built only from
        the LAST panel's handles -- if that panel happened to be missing a
        line for some algorithm (too few repetitions to compute a lag),
        that algorithm silently vanished from the legend even though it
        was plotted correctly in other panels. The legend is assembled
        from every panel, deduplicated, so nothing gets dropped.
        """
        if not groups:
            print(f"[plots] Skipping {filename}: no data.")
            return

        n_panels = len(groups)
        ncols = min(ncols, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows),
                                 sharey=True, squeeze=False)
        axes_flat = axes.flatten()

        for ax, group_value in zip(axes_flat, groups):
            group_df = self.pdf[self.pdf[group_col] == group_value]
            for alg in (self.algorithms or [None]):
                subset = (
                    group_df[group_df["algorithm"] == alg] if alg is not None
                    else group_df
                ).sort_values("run_id")
                series = subset["entropy_encrypted"].to_numpy()
                n = len(series)
                lag_limit = min(max_lag, n - 2)
                if lag_limit < 1:
                    continue
                series = series - series.mean()
                denom = np.dot(series, series)
                lags = np.arange(0, lag_limit + 1)
                acf = [
                    np.dot(series[:n - lag], series[lag:]) / denom if denom > 0 else 0.0
                    for lag in lags
                ]
                ax.plot(lags, acf, marker="o", markersize=2.5, linewidth=1, label=alg)

            ci = 1.96 / np.sqrt(max(len(group_df), 1))
            ax.axhspan(-ci, ci, color="gray", alpha=0.2)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(str(group_value), fontsize=9)
            ax.set_xlabel(f"Lag ({lag_unit})")

        # Hide any unused trailing axes (when n_panels isn't a multiple of ncols)
        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)

        for row in range(nrows):
            axes[row, 0].set_ylabel("Autocorrelation")

        # Build the legend from ALL panels, deduplicated, so an algorithm
        # missing from one panel (too few points there) doesn't get
        # silently dropped from the legend.
        handles, labels = [], []
        for ax in axes_flat[:n_panels]:
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi)
                    labels.append(li)
        if handles:
            fig.legend(handles, labels, loc="upper right",
                      bbox_to_anchor=(0.99, 0.99), fontsize=8)

        plt.suptitle(title, y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}.png", dpi=350)
        plt.close()

    def entropy_autocorrelation_per_file(self, top_n_files=None, safety_cap=30):
        """
        Controlled version of entropy_autocorrelation(): since every loop
        re-processes the SAME file list in the SAME order, run_id-based
        autocorrelation is confounded by file identity (file X's own
        entropy characteristics repeating every M runs, M = file count) --
        that confound alone can produce exactly the kind of decaying,
        above-the-band ACF you'd wrongly blame on the key generator.

        This isolates the key/IV generator instead: for each distinct
        file (or, if there are more than `safety_cap` of them, the
        `safety_cap` busiest by repetition count -- individual files can
        run into the hundreds, unlike extensions, so an unbounded grid
        here could get unreasonably large), compute the ACF of ITS OWN
        entropy sequence across loops only. Same file, same size,
        different key/IV each time -- if THIS collapses into the
        white-noise band while the run_id-based ACF didn't, the earlier
        correlation was a loop-structure artifact, not a cryptographic
        weakness.

        Pass top_n_files explicitly to force a smaller/larger selection.
        """
        counts = self.pdf["file_name"].value_counts()
        limit = top_n_files if top_n_files is not None else safety_cap
        if top_n_files is None and len(counts) > safety_cap:
            print(f"[plots] entropy_autocorrelation_per_file: {len(counts)} distinct "
                  f"files found, showing the {safety_cap} busiest (pass "
                  f"top_n_files=N to change this).")
        files = counts.head(limit).index.tolist()
        self._acf_panel_grid(
            group_col="file_name", groups=files,
            filename="entropy_autocorrelation_per_file",
            title="Autocorrelation WITHIN a fixed file, across loops only\n"
                  "(controls for the same-file-every-loop confound)",
            lag_unit="loops"
        )

    def entropy_autocorrelation_per_extension(self, top_n_extensions=None):
        """
        Same control as entropy_autocorrelation_per_file(), but grouped by
        file EXTENSION instead of by individual file. Pools every file
        sharing an extension into one sequence (sorted by run_id), which
        generalizes better than picking a handful of specific files, and
        matches how the rest of the report already groups by extension
        (entropy_delta_extension, extension_statistics).
        """
        counts = self.pdf["extension"].value_counts()
        extensions = (
            counts.index.tolist() if top_n_extensions is None
            else counts.head(top_n_extensions).index.tolist()
        )
        self._acf_panel_grid(
            group_col="extension", groups=extensions,
            filename="entropy_autocorrelation_per_extension",
            title="Autocorrelation WITHIN a file extension, across runs\n"
                  "(controls for file-identity/size confound, grouped by type)",
            lag_unit="runs"
        )

    # ═════════════════════════════════════════════════════════════════════
    # NEW: statistical physics
    # ═════════════════════════════════════════════════════════════════════

    def entropy_order_parameter(self):
        """
        Frames 'how far from an ideal cipher' as an order parameter,
        borrowing statistical-physics / phase-transition language:
        psi = 0 means the 'disordered phase' (ciphertext indistinguishable
        from uniform noise, entropy = 8 bits); psi = 1 would mean the
        'ordered phase' (no randomization at all, entropy = 0). A good
        cipher should sit as close to psi = 0 as possible, for every file,
        every run.
        """
        df = self.pdf.copy()
        df["psi"] = 1.0 - (df["entropy_encrypted"] / 8.0)

        plt.figure(figsize=(8, 6))
        sns.violinplot(data=df, x="algorithm", y="psi", inner="quartile", palette="colorblind")
        plt.axhline(0.0, color="black", linestyle="--", linewidth=1, label="Disordered phase (ideal cipher)")
        plt.ylabel(r"Order parameter  $\psi = 1 - H_{enc}/8$")
        plt.xlabel("Algorithm")
        plt.title("Entropy Order Parameter by Algorithm")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "entropy_order_parameter.png", dpi=350)
        plt.close()

    def maxent_uniform_demo(self):
        """
        Statistical-mechanics-flavored demonstration of the maximum
        entropy principle: among all probability distributions over
        N=256 outcomes (byte values), the UNIFORM distribution has the
        highest Shannon entropy -- exactly the 'equal a priori
        probabilities' postulate of the microcanonical ensemble. We sweep
        a family of distributions that concentrate an increasing fraction
        of probability mass on a single outcome, and show entropy falls
        monotonically away from the uniform (theoretical maximum, 8 bits)
        as concentration increases. Each algorithm's observed mean entropy
        is marked for reference.
        """
        N = 256
        concentrations = np.linspace(0.0, 0.99, 60)
        entropies = []
        for c in concentrations:
            probs = np.full(N, (1 - c) / (N - 1))
            probs[0] = c + (1 - c) / N
            probs = probs / probs.sum()
            entropies.append(-np.sum(probs * np.log2(probs)))

        plt.figure(figsize=(9, 6))
        plt.plot(concentrations, entropies, color="black", linewidth=2)
        plt.axhline(8.0, color="gray", linestyle=":", linewidth=1, label="Max entropy (uniform, N=256)")

        means = self.pdf.groupby("algorithm")["entropy_encrypted"].mean()
        for alg, val in means.items():
            plt.axhline(val, linestyle="--", alpha=0.6, label=f"{alg} mean = {val:.3f}")

        plt.xlabel("Concentration of probability mass on one byte value")
        plt.ylabel("Entropy (bits)")
        plt.title("Maximum Entropy Principle: Uniform Distribution Maximizes H")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(self.output_dir / "maxent_uniform_demo.png", dpi=350)
        plt.close()

    # ═════════════════════════════════════════════════════════════════════

    def bit_entropy_plot(self):
        """
        Real H(p) vs p plot using the exact bit-level measurements from
        bit_entropy.py ('bernoulli_p' / 'bit_entropy' columns) -- not the
        normalized-byte-entropy approximation used by binary_entropy_curve().
        Skips gracefully on older CSVs that predate these columns.
        """
        required = {"bernoulli_p", "bit_entropy"}
        if not required.issubset(self.pdf.columns):
            print("[plots] Skipping bit_entropy_plot: CSV has no "
                  "'bernoulli_p'/'bit_entropy' columns (older run, or "
                  "runner.py not yet updated to record them).")
            return

        p_grid = np.linspace(0.0, 1.0, 400)
        h_grid = binary_entropy(p_grid)

        algorithms = self.algorithms if self.algorithms else ["data"]
        palette = sns.color_palette("colorblind", n_colors=len(algorithms))
        color_map = dict(zip(algorithms, palette))

        plt.figure(figsize=(10, 7.5))

        # Theoretical curve as background, with a light fill for depth
        plt.fill_between(p_grid, 0, h_grid, color="#d9d9d9", alpha=0.25, zorder=0)
        plt.plot(p_grid, h_grid, color="#1a1a1a", linewidth=2.2, zorder=1,
                  label="Curva teórica H(p)")
        plt.axvline(0.5, color="#888888", linestyle=":", linewidth=1.3, zorder=1,
                    label="p = 0.5 (máxima incertidumbre)")

        # Real measured points, one color per algorithm
        sns.scatterplot(
            data=self.pdf, x="bernoulli_p", y="bit_entropy", hue="algorithm",
            palette=color_map, alpha=0.6, s=55,
            edgecolor="white", linewidth=0.6, zorder=2
        )

        # Mean point per algorithm as a big star, so the "center of mass"
        # of each algorithm's cloud is easy to read at a glance
        means = self.pdf.groupby("algorithm")[["bernoulli_p", "bit_entropy"]].mean()
        for alg, row in means.iterrows():
            plt.scatter(
                row["bernoulli_p"], row["bit_entropy"], marker="*", s=380,
                color=color_map.get(alg, "black"), edgecolor="black",
                linewidth=0.9, zorder=3
            )

        plt.xlabel("Parámetro de Bernoulli real  p  (proporción de bits = 1)", fontsize=12)
        plt.ylabel("Entropía binaria real  H(p)  [bits]", fontsize=12)
        plt.title("H(p) vs p — medición exacta bit a bit (unpackbits)",
                  fontsize=14, weight="bold")
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.05)
        plt.grid(alpha=0.25)
        plt.legend(title="Algoritmo", frameon=True, loc="lower center",
                   ncol=max(len(algorithms), 1))
        plt.tight_layout()
        plt.savefig(self.output_dir / "binary_entropy_curve_real_bits.png", dpi=350)
        plt.close()

    # ═════════════════════════════════════════════════════════════════════
    # NEW: hash statistics (sha256_raw / sha256_encrypted)
    # ═════════════════════════════════════════════════════════════════════

    def hash_collision_check(self):
        """
        Flags any duplicate sha256_encrypted value across DIFFERENT
        experiment_id rows. For a correct SHA-256 over genuinely distinct
        ciphertexts this is astronomically unlikely (birthday bound), so
        a real duplicate here is a strong signal of an actual bug --
        e.g. two runs somehow producing byte-identical ciphertext, or a
        hashing/storage bug in the pipeline -- not a property of SHA-256
        itself, which needs no further statistical justification.
        """
        dupes = self.pdf[self.pdf.duplicated(subset=["sha256_encrypted"], keep=False)]
        with open(self.output_dir / "hash_collision_report.txt", "w") as f:
            f.write("SHA-256 ciphertext-hash collision check\n")
            f.write("=" * 45 + "\n\n")
            if dupes.empty:
                f.write(f"No sha256_encrypted collisions found across {len(self.pdf)} rows.\n")
            else:
                f.write(f"WARNING: {len(dupes)} rows share a duplicate "
                        f"sha256_encrypted value.\n\n")
                f.write(dupes[["experiment_id", "algorithm", "file_name",
                              "sha256_encrypted"]].to_string(index=False))
        return dupes

    def hash_byte_uniformity(self):
        """
        Chi-square goodness-of-fit test on the pooled bytes of
        sha256_raw and sha256_encrypted, by algorithm, against H0:
        'every byte value (0-255) is equally likely'. This is less a
        test of the ciphers than a sanity check on the measurement
        pipeline itself: SHA-256 output is designed to be statistically
        indistinguishable from uniform noise, so a failure here would
        point at a bug in hashing, hex-encoding, or storage -- not at
        SHA-256, which is already extensively studied elsewhere.
        """
        results = []
        for alg, group in self.pdf.groupby("algorithm"):
            for col, label in [("sha256_raw", "raw"), ("sha256_encrypted", "encrypted")]:
                hex_strings = group[col].dropna().astype(str)
                hex_strings = hex_strings[hex_strings.str.len() == 64]
                if hex_strings.empty:
                    continue
                all_bytes = b"".join(bytes.fromhex(h) for h in hex_strings)
                counts = np.bincount(np.frombuffer(all_bytes, dtype=np.uint8), minlength=256)
                expected = np.full(256, counts.sum() / 256)
                stat, p = chisquare(counts, expected)
                results.append({
                    "algorithm": alg, "hash_type": label,
                    "total_hash_bytes": int(counts.sum()),
                    "chi2_stat": stat, "p_value": p
                })
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / "hash_chi_square_uniformity.csv", index=False)
        return results_df

    def hash_avalanche_effect(self, sample_files=6, max_pairs_per_file=2000):
        """
        For the busiest repeated files, samples pairs of
        sha256_encrypted values across different loops (same file,
        different key/IV each run) and computes the fraction of
        DIFFERING BITS between each pair's hash. An ideal cipher's
        avalanche effect means ~50% of ciphertext (and therefore hash)
        bits should differ between any two independent runs, even on
        the exact same input file. Plots that fraction's distribution
        against the 50% ideal.
        """
        top_files = self.pdf["file_name"].value_counts().head(sample_files).index.tolist()
        rng = np.random.default_rng(42)
        fractions = []

        for fname in top_files:
            hashes = self.pdf.loc[self.pdf["file_name"] == fname, "sha256_encrypted"].dropna()
            hashes = [h for h in hashes if len(h) == 64]
            n = len(hashes)
            if n < 2:
                continue
            n_pairs = min(max_pairs_per_file, n * (n - 1) // 2)
            idx = rng.integers(0, n, size=(n_pairs, 2))
            for i, j in idx:
                if i == j:
                    continue
                b1 = bytes.fromhex(hashes[i])
                b2 = bytes.fromhex(hashes[j])
                xor = int.from_bytes(b1, "big") ^ int.from_bytes(b2, "big")
                diff_bits = bin(xor).count("1")
                fractions.append(diff_bits / (len(b1) * 8))

        if not fractions:
            print("[plots] Skipping hash_avalanche_effect: not enough repeated-file data.")
            return

        plt.figure(figsize=(9, 6))
        sns.histplot(fractions, bins=40, kde=True, color="#2c7bb6")
        plt.axvline(0.5, color="black", linestyle="--", linewidth=1.5,
                    label="Ideal avalanche (50% of bits differ)")
        mean_frac = float(np.mean(fractions))
        plt.axvline(mean_frac, color="#d7191c", linestyle=":", linewidth=1.5,
                    label=f"Observed mean = {mean_frac:.4f}")
        plt.xlabel("Fraction of differing bits between two ciphertext hashes\n"
                  "(same file, different key/IV each run)")
        plt.ylabel("Count")
        plt.title("Avalanche Effect: Hash Bit-Difference Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "hash_avalanche_effect.png", dpi=350)
        plt.close()

    def generate_all(self):
        self.entropy_histogram()
        self.entropy_boxplot()
        self.before_after_entropy()
        self.entropy_delta_boxplot()
        self.entropy_delta_extension()
        self.entropy_delta_vs_time()
        self.entropy_delta_ecdf()
        self.extension_statistics()
        self.entropy_summary()
        self.entropy_violin_advanced()
        self.entropy_delta()
        self.encryption_times()
        self.decryption_times()
        self.entropy_scatter()
        self.entropy_ecdf()
        self.time_vs_entropy_scatter()
        self.pairwise_entropy_heatmap()
        self.combined_dashboard()
        self.mean_error_by_algorithm()

        # H(p) vs p
        self.binary_entropy_curve()
        self.binary_entropy_curve_by_algorithm()
        self.bit_entropy_plot()
        self.entropy_vs_size_ceiling()

        # Distribution diagram
        self.distribution_diagram()

        # Convexity / concavity
        self.entropy_concavity_demo()
        self.relative_entropy_convexity_demo()

        # Stochastic processes
        self.entropy_stochastic_trajectory()
        self.entropy_autocorrelation()
        self.entropy_autocorrelation_per_file()
        self.entropy_autocorrelation_per_extension()

        # Statistical physics
        self.entropy_order_parameter()
        self.maxent_uniform_demo()

        # Hash statistics (sha256_raw / sha256_encrypted)
        self.hash_collision_check()
        self.hash_byte_uniformity()
        self.hash_avalanche_effect()

        # statistics
        self.descriptive_statistics()
        self.wilcoxon_per_algorithm()
        self.kruskal_wallis_test()
        self.wilcoxon_test()

        print("All professional-grade plots and statistics generated successfully.")