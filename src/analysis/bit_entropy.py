"""
bit_entropy.py — Bit-level (not byte-level) entropy measurement.

Unlike src.analysis.entropy.shannonEntropy (which works over the 256-symbol
byte alphabet, max 8 bits/byte), this module treats the file as a raw
stream of individual bits and computes the exact Bernoulli parameter p
(fraction of bits equal to 1) and the binary Shannon entropy H(p), in bits.
"""

from pathlib import Path
import numpy as np


def compute_bit_level_entropy(file_path, chunk_size=4 * 1024 * 1024):
    """
    Streams `file_path` in blocks of `chunk_size` bytes (default 4 MiB) so
    memory use stays bounded regardless of file size -- safe for large
    video files. Each block is expanded into its real, individual bits
    with numpy.unpackbits (exact bit-by-bit unpacking, not an estimate),
    and two running counters (bits seen, bits equal to 1) are updated;
    nothing but those two integers survives past each block.

    Returns
    -------
    (p, h) : tuple[float, float]
        p -- Bernoulli parameter: exact fraction of bits equal to 1 over
             the whole file.
        h -- binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p), in bits.
             p is clipped away from the exact boundaries {0, 1} before the
             logarithms, so a block of pure zeros (p=0) or pure ones (p=1)
             -- a completely valid, real outcome for a weak cipher -- still
             yields a well-defined H = 0.0 instead of a log(0) / NaN.

    Raises
    ------
    RuntimeError
        If the file cannot be opened or read (wraps the underlying OSError
        with the offending path for a clearer traceback).
    """
    path = Path(file_path)
    total_bits = 0
    ones = 0

    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                byte_array = np.frombuffer(chunk, dtype=np.uint8)
                bits = np.unpackbits(byte_array)  # exact 0/1 expansion
                ones += int(bits.sum())
                total_bits += bits.size
    except OSError as exc:
        raise RuntimeError(
            f"compute_bit_level_entropy: could not read '{path}': {exc}"
        ) from exc

    if total_bits == 0:
        # Empty file: nothing to measure. Report p=0.0, H=0.0 by
        # convention instead of raising a ZeroDivisionError.
        return 0.0, 0.0

    p = ones / total_bits

    # Clip strictly inside (0, 1) so log2(p) / log2(1-p) never hit log(0).
    p_safe = min(max(p, 1e-12), 1 - 1e-12)
    h = -p_safe * np.log2(p_safe) - (1 - p_safe) * np.log2(1 - p_safe)

    return p, float(h)