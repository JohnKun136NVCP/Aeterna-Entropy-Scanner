"""
key_statistics.py — Statistical analysis of the cryptographic keys and
IVs themselves (not the ciphertext), reading the *_keys.csv files.

Every test here works directly on RAW key/IV bytes: whether the KEY
GENERATOR is producing statistically sound randomness, independent of
how the cipher that later consumes it behaves. This is exactly the kind
of test that would have caught the DES ASCII-restricted key bug in one
shot, instead of tracing it back indirectly through low ciphertext
entropy.
"""

from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare, binom

sns.set_theme(style="whitegrid")


def _hex_to_bytes(hex_str):
    if not isinstance(hex_str, str) or hex_str == "":
        return b""
    try:
        return bytes.fromhex(hex_str)
    except ValueError:
        return b""


class KeyStatistics:

    def __init__(self, csv_file, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_files = self._resolve_csv_files(csv_file)
        if not self.csv_files:
            raise FileNotFoundError(f"No keys CSV matched: {csv_file}")

        frames = [pd.read_csv(p) for p in self.csv_files]
        self.df = pd.concat(frames, ignore_index=True)

        self.df["key_bytes"] = self.df["key"].apply(_hex_to_bytes)
        self.df["iv_bytes"] = self.df["iv"].apply(_hex_to_bytes)
        self.algorithms = sorted(self.df["algorithm"].dropna().unique())

    @staticmethod
    def _resolve_csv_files(csv_file):
        """Same convention as PlotGenerator: single path, list, or glob."""
        if isinstance(csv_file, (list, tuple)):
            candidates = [Path(p) for p in csv_file]
        else:
            csv_file = str(csv_file)
            if any(ch in csv_file for ch in "*?[]"):
                candidates = [Path(p) for p in sorted(glob.glob(csv_file))]
            else:
                candidates = [Path(csv_file)]
        return [p for p in candidates if p.exists()]

    # ═════════════════════════════════════════════════════════════════════
    # 1. Chi-square byte uniformity
    # ═════════════════════════════════════════════════════════════════════

    def chi_square_byte_uniformity(self):
        """
        Pools every byte from every key of a given algorithm into one
        sample, builds a 256-bin histogram, and runs a chi-square
        goodness-of-fit test against H0: 'every byte value (0-255) is
        equally likely'. The standard test for whether a key generator
        genuinely uses the full byte range. A structural bias (e.g. an
        alphabet-restricted generator) shows up as either a significant
        chi-square statistic, or, more bluntly, as byte values with
        literally zero count that should be common under H0.
        """
        results = []
        for alg, group in self.df.groupby("algorithm"):
            all_bytes = b"".join(group["key_bytes"])
            if not all_bytes:
                continue
            counts = np.bincount(np.frombuffer(all_bytes, dtype=np.uint8), minlength=256)
            expected = np.full(256, counts.sum() / 256)
            stat, p = chisquare(counts, expected)
            results.append({
                "algorithm": alg,
                "total_key_bytes": int(counts.sum()),
                "chi2_stat": stat,
                "p_value": p,
                "zero_count_byte_values": int((counts == 0).sum())
            })

            plt.figure(figsize=(10, 5))
            plt.bar(np.arange(256), counts, width=1.0, color="#2c7bb6")
            plt.axhline(counts.sum() / 256, color="black", linestyle="--",
                        label="Expected under uniform")
            plt.xlabel("Byte value (0-255)")
            plt.ylabel("Count across all keys")
            verdict = "REJECTS uniformity (p<0.05)" if p < 0.05 else "consistent with uniform"
            plt.title(f"Key byte-value histogram -- {alg}\n"
                      f"chi2={stat:.1f}, p={p:.3g}  ({verdict})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / f"key_byte_uniformity_{alg}.png", dpi=350)
            plt.close()

        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / "key_chi_square_uniformity.csv", index=False)
        return results_df

    # ═════════════════════════════════════════════════════════════════════
    # 2. Hamming weight vs. theoretical binomial
    # ═════════════════════════════════════════════════════════════════════

    def hamming_weight_distribution(self):
        """
        For each key, counts how many of its bits are 1 (Hamming weight).
        For a truly random n-bit key, this follows Binomial(n, 0.5).
        Plots the empirical histogram against the theoretical binomial
        PMF, per algorithm -- the same 'theoretical vs experimental'
        spirit as the ciphertext entropy plots, applied to the keys.
        """
        for alg, group in self.df.groupby("algorithm"):
            valid = group["key_bytes"][group["key_bytes"].str.len() > 0]
            if valid.empty:
                continue
            n_bits = len(valid.iloc[0]) * 8
            weights = valid.apply(lambda b: bin(int.from_bytes(b, "big")).count("1"))

            plt.figure(figsize=(9, 6))
            bins = np.arange(0, n_bits + 2) - 0.5
            plt.hist(weights, bins=bins, density=True, alpha=0.6,
                    color="#2c7bb6", label="Observed")

            x = np.arange(0, n_bits + 1)
            plt.plot(x, binom.pmf(x, n_bits, 0.5), color="#d7191c", linewidth=2,
                    label=f"Binomial({n_bits}, 0.5) theoretical")

            plt.xlabel("Hamming weight (number of 1-bits per key)")
            plt.ylabel("Density")
            plt.title(f"Key Hamming Weight Distribution -- {alg}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / f"key_hamming_weight_{alg}.png", dpi=350)
            plt.close()

    # ═════════════════════════════════════════════════════════════════════
    # 3. Monobit test by bit position
    # ═════════════════════════════════════════════════════════════════════

    def bit_position_monobit(self):
        """
        For each bit position (0 = MSB of the first byte), the fraction
        of keys with a 1 there. Should hover around 0.5 everywhere. A
        position stuck near 0 or 1 across ALL keys is a structural bias
        in the key generator -- exactly what the ASCII-restricted DES
        generator produced (bit 7 of every byte fixed at 0, i.e. that
        position would show 0.0 here instead of ~0.5).
        """
        for alg, group in self.df.groupby("algorithm"):
            key_list = [b for b in group["key_bytes"] if b]
            if not key_list:
                continue
            n_bytes = len(key_list[0])
            same_len = [b for b in key_list if len(b) == n_bytes]
            bit_arrays = np.array([
                np.unpackbits(np.frombuffer(b, dtype=np.uint8)) for b in same_len
            ])
            fractions = bit_arrays.mean(axis=0)

            plt.figure(figsize=(12, 5))
            plt.bar(np.arange(len(fractions)), fractions, color="#2c7bb6")
            plt.axhline(0.5, color="black", linestyle="--", label="Expected (0.5)")
            ci = 1.96 * np.sqrt(0.25 / max(len(bit_arrays), 1))
            plt.axhspan(0.5 - ci, 0.5 + ci, color="gray", alpha=0.2, label="~95% CI")
            plt.xlabel("Bit position (0 = MSB of first byte)")
            plt.ylabel("Fraction of keys with 1 at this position")
            plt.title(f"Monobit Test by Bit Position -- {alg}")
            plt.ylim(0, 1)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / f"key_monobit_by_position_{alg}.png", dpi=350)
            plt.close()

    # ═════════════════════════════════════════════════════════════════════
    # 4. Key/IV reuse check
    # ═════════════════════════════════════════════════════════════════════

    def key_iv_reuse_check(self):
        """
        Flags any (key, iv) pair used more than once for the same
        algorithm. Reusing a key+IV pair in CBC mode is a real, known
        vulnerability (it reveals the XOR of the two plaintexts). Writes
        a text report; an empty/clean report is the good outcome.
        """
        report_lines = []
        any_reuse = False
        for alg, group in self.df.groupby("algorithm"):
            pair = group["key"].astype(str) + "|" + group["iv"].fillna("").astype(str)
            dup_mask = pair.duplicated(keep=False)
            if dup_mask.any():
                any_reuse = True
                dup_ids = group.loc[dup_mask, "experiment_id"].tolist()
                shown = dup_ids[:20]
                more = " ..." if len(dup_ids) > 20 else ""
                report_lines.append(
                    f"{alg}: {int(dup_mask.sum())} filas con (key, iv) repetido -- "
                    f"experiment_id: {shown}{more}"
                )
            else:
                report_lines.append(
                    f"{alg}: sin repeticiones de (key, iv) en {len(group)} filas."
                )

        with open(self.output_dir / "key_iv_reuse_report.txt", "w") as f:
            f.write("Key/IV reuse check\n")
            f.write("=" * 40 + "\n\n")
            f.write("\n".join(report_lines))
            f.write("\n\nResultado: " + (
                "SE ENCONTRARON REPETICIONES -- revisar."
                if any_reuse else
                "sin repeticiones detectadas."
            ))

        return any_reuse

    # ═════════════════════════════════════════════════════════════════════
    # 5. Key bit-level entropy, on the same H(p) reference curve
    # ═════════════════════════════════════════════════════════════════════

    def key_bit_entropy(self):
        """
        Exact Bernoulli parameter and binary entropy computed directly
        from key bytes (no disk I/O needed -- keys are already in
        memory), plotted against the same theoretical H(p) curve used
        for the ciphertext analysis, so key quality and ciphertext
        quality sit on directly comparable axes.
        """
        from src.visualization.plots import (
            binary_entropy, sample_for_scatter
        )

        records = []
        for _, row in self.df.iterrows():
            b = row["key_bytes"]
            if not b:
                continue
            bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
            p = float(bits.mean())
            p_safe = min(max(p, 1e-12), 1 - 1e-12)
            h = -p_safe * np.log2(p_safe) - (1 - p_safe) * np.log2(1 - p_safe)
            records.append({"algorithm": row["algorithm"], "p": p, "h": h})

        if not records:
            print("[key_statistics] key_bit_entropy: no key bytes available.")
            return

        plot_df = sample_for_scatter(pd.DataFrame(records), max_per_group=4000)

        # Small per-algorithm jitter, same reasoning as the ciphertext
        # H(p) plot: identical/near-identical key generators would
        # otherwise bury each other visually.
        algos_present = sorted(plot_df["algorithm"].dropna().unique())
        step = 0.006
        jitter_map = {
            alg: (i - (len(algos_present) - 1) / 2) * step
            for i, alg in enumerate(algos_present)
        }
        plot_df["p_jittered"] = plot_df["p"] + plot_df["algorithm"].map(jitter_map).fillna(0)

        p_grid = np.linspace(0.0, 1.0, 400)
        h_grid = binary_entropy(p_grid)

        plt.figure(figsize=(9, 7))
        plt.plot(p_grid, h_grid, color="black", linewidth=2, label="H(p) theoretical curve")
        plt.axvline(0.5, color="gray", linestyle=":", label="p = 0.5 (ideal key)")
        sns.scatterplot(data=plot_df, x="p_jittered", y="h", hue="algorithm", alpha=0.6, s=30)
        plt.xlabel("Bernoulli parameter p of the KEY bits (tiny offset for visibility)")
        plt.ylabel("Binary entropy H(p) of the KEY  [bits]")
        plt.title("Key Quality: H(p) vs p (computed directly from key bytes)")
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.05)
        plt.legend(title="Algorithm", loc="lower center")
        plt.tight_layout()
        plt.savefig(self.output_dir / "key_bit_entropy_curve.png", dpi=350)
        plt.close()

    # ═════════════════════════════════════════════════════════════════════

    def generate_all(self):
        self.chi_square_byte_uniformity()
        self.hamming_weight_distribution()
        self.bit_position_monobit()
        self.key_iv_reuse_check()
        self.key_bit_entropy()
        print("Key/IV statistical analysis complete.")