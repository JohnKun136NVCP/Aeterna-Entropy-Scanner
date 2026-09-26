# CipherEntropy

[![CipherEntropy Logo](https://github.com/JohnKun136NVCP/CipherEntropy/raw/main/img/frieren_ascii.png)](/JohnKun136NVCP/CipherEntropy/blob/main/img/frieren_ascii.png)

Created by [JohnKun136NVCP](https://github.com/JohnKun136NVCP)

Project to get my bachelor's degree in physics and computer science, and to learn more about cryptography and entropy analysis.

## Project Overview

CipherEntropy is a cryptographic research tool that measures and statistically characterizes the randomness of encrypted data. It runs a configurable batch of files through **AES-CBC**, **DES-CBC**, and **RC4**, and for every run records both the classic byte-level Shannon entropy of the ciphertext and a set of deeper, complementary analyses: exact bit-level entropy (computed with `numpy.unpackbits`, not estimated from byte frequencies), the encryption/decryption keys and IVs themselves, and cryptographic hashes of both the plaintext and ciphertext.

Beyond the raw entropy number, the project treats the problem as one of information theory and statistics rather than a single pass/fail metric:

- **Information theory**: every measured entropy value is placed on the theoretical binary entropy curve *H(p)*, with the achievable ceiling correctly adjusted for each file's size (a byte-level entropy estimate can never exceed `log2(min(size, 256))` bits, and even an ideal cipher's *expected* value for small samples sits below the flat 8-bit ideal by a known, closed-form bias — the Miller–Madow correction). Convexity of relative entropy (KL divergence) and concavity of Shannon entropy are demonstrated directly against the experiment's own data.
- **Statistics**: descriptive statistics, Wilcoxon signed-rank tests (paired, raw vs. encrypted), Mann-Whitney U with Bonferroni correction for pairwise algorithm comparisons, and Kruskal-Wallis across all three ciphers.
- **Stochastic processes**: the sequence of entropy measurements is treated as a sample path, checked for (weak) stationarity via rolling statistics, and tested for serial independence via autocorrelation — including controlled versions that isolate the key generator from the confound of the experiment re-processing the same file list every loop.
- **Statistical physics**: an entropy-based "order parameter" framing analogous to phase transitions, and a maximum-entropy-principle demonstration showing why the uniform byte distribution is the correct theoretical ceiling.
- **Key/IV and hash statistics**: independent of ciphertext quality, the keys and IVs themselves are tested for byte-level uniformity (chi-square goodness-of-fit), Hamming-weight distribution against the theoretical binomial, a monobit test by bit position (the test that directly exposes a structurally biased key generator), key/IV reuse detection, and hash-based checks (collision check, byte uniformity, avalanche effect between repeated encryptions of the same file).

All of this is designed to be safe to run repeatedly and in bulk: every file in the input directory is processed on an isolated temporary copy, so the original data is never modified, moved, or deleted, even if a run crashes partway through.

## Features

- **Entropy Analysis**: Shannon entropy (byte-level) and exact binary entropy (bit-level, via `numpy.unpackbits`) for both original and encrypted data, with a size-corrected theoretical ceiling
- **Cipher Support**: AES-CBC, DES-CBC (custom C implementation with a table-driven fast path), and RC4
- **Statistical Analysis**: descriptive stats, non-parametric hypothesis tests (Wilcoxon, Mann-Whitney, Kruskal-Wallis), autocorrelation, chi-square uniformity, Hamming-weight/monobit tests on keys, hash collision/avalanche checks
- **Information-Theory Visuals**: H(p) vs. p curves, concavity/convexity demonstrations, maximum-entropy-principle plots, entropy-as-order-parameter plots
- **CLI Integration**: flexible command-line interface with persisted configuration (`config.json`)
- **Safe by Design**: every cipher operation runs on a temporary copy; the original dataset is never touched
- **Parallelized**: file-level multithreading (real parallelism across cores, since the C extensions release the GIL during encryption/decryption)
- **Multiple Output Formats**: per-algorithm or combined CSV datasets, high-resolution PNG figures

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Compile the C extensions (DES and RC4) before running — see [Makefile Compilation](#makefile-compilation) below.

## Usage

### Basic Execution

```bash
python3 main.py -r
```

### CLI Configuration Options

Every option is persisted to `config.json` after each run, so omitted flags fall back to your last-used values (or the defaults below) rather than resetting each time.

| Flag | Description | Default |
|---|---|---|
| `-n` | Run with `n` loops | `1` |
| `-al` | Algorithm to run: `DES`, `RC4`, `AES`, or `All` | `All` |
| `-i` | Show the ASCII art prompt (`yes`/`no`/`true`/`false`/`1`/`0`) | show |
| `-sd` | Directory of data to encrypt | `./data` |
| `-sp` | Directory to save plots | `./data/plots` |
| `-rc` | Reset `config.json` to defaults | `False` |
| `-op` | Only generate plots/statistics; skip the cipher run | `False` |
| `-np` | Skip plot generation entirely | `False` |
| `-oc` | Base name for the output CSV files (e.g. `aes` → `aes.csv` / `aes_keys.csv`) | `global` |
| `-cp` | Combine every `data/csv/*.csv` file for cross-algorithm plots, instead of just this job's own CSV | `False` |
| `-ka` | Also run the key/IV and hash statistical analysis (`KeyStatistics`) | `False` |
| `-r` | Run the program (required to actually execute anything) | `False` |

## Example Commands

```bash
# Basic entropy analysis: all three algorithms, 1000 loops
python3 main.py -n 1000 -al All -r

# Run a single algorithm into its own CSV, no ASCII prompt, no plots yet
python3 main.py -al AES -i false -np -oc aes -n 200 -r
python3 main.py -al DES -i false -np -oc des -n 200 -r
python3 main.py -al RC4 -i false -np -oc rc4 -n 200 -r

# Generate plots afterwards, combining all three algorithms, plus key/hash statistics
python3 main.py -i false -op -cp -ka -sp data/plots -r

# Run a large job in the background (survives closing the terminal)
nohup python3 main.py -al AES -i false -np -oc aes -n 5000 -r > aes_job.log 2>&1 &

# Reset the configuration to default
python3 main.py -rc True
```

## Project Structure

```
CipherEntropy/
├── main.py                      # Entry point -> calls cli.main()
├── cli.py                       # Argument parsing, config.json persistence
├── config.py                    # ROOT path and shared constants
├── ascii.txt                    # ASCII art shown by -i
├── requirements.txt
├── Makefile                     # Builds libdes.so / librc4.so
├── LICENSE
│
├── src/
│   ├── crypto/                  # Cipher implementations
│   │   ├── des.c / des.h        # DES-CBC core (table-driven fast path)
│   │   ├── des_tables.c / .h    # DES permutation/S-box tables
│   │   ├── des.py               # ctypes wrapper around libdes.so
│   │   ├── rc4.c / rc4.h        # RC4 stream cipher core
│   │   ├── rc4.py               # ctypes wrapper around librc4.so
│   │   ├── aes.py               # AES-CBC via the `cryptography` library
│   │   ├── shaCipher.py         # SHA-256 file hashing helper
│   │   └── randomness.py        # Cryptographically secure key/IV generation
│   │
│   ├── analysis/                # Measurement and data layer
│   │   ├── entropy.py           # Byte-level Shannon entropy
│   │   ├── bit_entropy.py       # Exact bit-level entropy (numpy.unpackbits)
│   │   ├── frecuency.py         # Byte-frequency helpers
│   │   ├── dataset.py           # CSV schema + append/read logic
│   │   └── key_statistics.py    # Key/IV/hash statistical tests
│   │
│   ├── visualization/
│   │   └── plots.py             # PlotGenerator: every figure and statistical test
│   │
│   └── experiments/
│       └── runner.py            # Orchestrates one full cipher/loop/file run
│
├── data/
│   ├── csv/                     # <name>.csv (experiments) + <name>_keys.csv
│   │   └── ...                  # never modified in place; each row is appended
│   └── plots/                   # Generated figures and statistics reports
│
├── notebooks/                   # Exploratory analysis notebooks
└── img/                         # README assets
```

**Data flow, in short**: `cli.py` reads flags and `config.json` → `runner.py` copies each input file into an isolated temp directory, encrypts/decrypts it there with the chosen cipher(s), measures entropy (byte- and bit-level) and hashes, and appends one row per run to `dataset.py`'s CSV → `plots.py` (`PlotGenerator`) and `key_statistics.py` (`KeyStatistics`) read those CSVs back and produce every figure and statistical report under `data/plots/`.

## Hex Representation

```
63 69 70 68 65 72 65 6e 74 72 6f 70 79
```

## Results Interpretation

### Shannon Entropy (byte-level)

Values range from 0 to 8 bits/byte:

- **0–2**: Low entropy (non-random, weak encryption)
- **2–6**: Moderate entropy (acceptable encryption)
- **6–8**: High entropy (strong encryption, good randomness)

**Important caveat**: this ceiling of 8 bits/byte is only reachable for files large enough to contain close to 256 distinct byte values. A byte-level entropy estimate computed over *N* bytes can never exceed `log2(min(N, 256))` — a 13-byte file cannot score above ~3.7 bits no matter how strong the cipher is. `PlotGenerator` accounts for this with a size-adjusted normalization and a Miller–Madow bias-corrected reference curve; always read small-file results against that ceiling, not the flat 8-bit line.

### Bit-level Entropy

Computed independently from byte-level entropy, via `bit_entropy.py`, by unpacking every byte into its individual bits and measuring the exact Bernoulli parameter *p* (fraction of bits equal to 1) and binary entropy *H(p)*. This is a stricter, alphabet-independent check — it doesn't inherit the byte-alphabet ceiling above, since a file only needs to be a handful of bytes to already contain hundreds of independent bit samples.

### Configuration Example

```json
{
  "loops": 1,
  "algo": "All",
  "show": true,
  "savedData": "/path/to/data",
  "savedPlot": "/path/to/plots",
  "onplots": false,
  "noPlots": false,
  "csvName": "global",
  "combinePlots": false,
  "keyAnalysis": false
}
```

## Requirements

- Python 3.10+
- GCC (to compile the DES and RC4 C extensions)
- `cryptography` (AES)
- NumPy (bit-level entropy, statistical calculations)
- Matplotlib / Seaborn (plotting)
- SciPy (hypothesis tests, numerical root-finding)
- Pandas (data manipulation)
- statannotations (pairwise significance annotations)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please submit pull requests or issues to the project repository.

## Support

For issues, questions, or suggestions, please open an issue on the project repository. Connect with the telegram that is available on my github profile.

## Makefile Compilation

Add the following commands to your Makefile:

```
gcc -O2 -Wall -fPIC -c des.c -o des.o
gcc -O2 -Wall -fPIC -c des_tables.c -o des_tables.o
gcc -shared -o libdes.so des.o des_tables.o
gcc -shared -fPIC rc4.c -o librc4.so
```