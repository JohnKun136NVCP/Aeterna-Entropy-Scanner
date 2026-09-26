import os
import csv
import time
import shutil
import tempfile
import concurrent.futures
from config import ROOT
from src.analysis.entropy import shannonEntropy
from src.analysis.bit_entropy import compute_bit_level_entropy
from src.visualization.plots import PlotGenerator
from src.analysis.key_statistics import KeyStatistics
from src.analysis.frecuency import byte_frequency, byte_frequency_normalized
from src.analysis.dataset import Dataset
from src.crypto.des import DES
from src.crypto.rc4 import rc4_file
from src.crypto.aes import aes_encrypt_file, aes_decrypt_file
from src.crypto.shaCipher import sha256_file
from src.crypto.randomness import (
    generateTokenCiphers,
    generate_iv
)

# Run DES
des = DES()

# Extensions the pipeline could leave behind if something ever failed.
# Ignored when listing the data folder so leftovers from previous runs
# never get re-encrypted by accident.
_INTERMEDIATE_EXTENSIONS = (".enc", ".rc4", ".aes", ".dec")


def get_files(directory):
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
        and not f.endswith(_INTERMEDIATE_EXTENSIONS)
    ]
    return sorted(files)


def generate_key_iv(cipher: str) -> tuple:
    if cipher == "DES":
        return generateTokenCiphers(cipher), generate_iv(cipher)
    elif cipher == "RC4":
        return generateTokenCiphers(cipher), None
    else:
        return generateTokenCiphers(cipher), generate_iv(cipher)


def to_hex(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def run_once(file_path: str, cipher: str, run_id: int):
    """
    All encrypt/decrypt work happens on a COPY of the file inside a
    temporary directory. The original file in data/ is only ever opened
    for reading (shutil.copy2) and is never written to, moved, or deleted.

    The temp directory is destroyed as soon as the `with` block exits,
    even if an exception is raised halfway through, so no .enc/.rc4/.aes
    leftover can ever end up in data/ to "poison" the next run.
    """
    original_name = os.path.basename(file_path)

    with tempfile.TemporaryDirectory(prefix="cipherentropy_") as tmp_dir:
        work_path = os.path.join(tmp_dir, original_name)
        shutil.copy2(file_path, work_path)  # copy only, original is untouched

        with open(work_path, "rb") as f:
            data = f.read()

        key, iv = generate_key_iv(cipher)

        entropy_raw = shannonEntropy(data)
        freq_raw = byte_frequency(data)
        sha_raw = sha256_file(work_path)

        suffix = {"DES": ".enc", "RC4": ".rc4", "AES": ".aes"}[cipher]
        enc_path = work_path + suffix

        if cipher == "DES":
            # A fresh instance per call: DES() holds mutable key/subkey
            # state, so sharing the module-level `des` object across
            # threads would let concurrent runs clobber each other's key.
            local_des = DES()
            local_des.randKey(key)
            local_des.read_iv_file(work_path)
            start = time.perf_counter()
            local_des.encrypt_file(work_path)
            encrypt_time = (time.perf_counter() - start) * 1000
            with open(enc_path, "rb") as f:
                ciphertext = f.read()
            iv = local_des.read_iv_file(enc_path)
            sha_enc = sha256_file(enc_path)
            start = time.perf_counter()
            local_des.decrypt_file(enc_path)
            decrypt_time = (time.perf_counter() - start) * 1000

        elif cipher == "RC4":
            start = time.perf_counter()
            rc4_file(work_path, enc_path, key)
            encrypt_time = (time.perf_counter() - start) * 1000
            with open(enc_path, "rb") as f:
                ciphertext = f.read()
            sha_enc = sha256_file(enc_path)
            start = time.perf_counter()
            rc4_file(enc_path, work_path, key)
            decrypt_time = (time.perf_counter() - start) * 1000

        elif cipher == "AES":
            start = time.perf_counter()
            aes_encrypt_file(work_path, enc_path, key, iv)
            encrypt_time = (time.perf_counter() - start) * 1000
            with open(enc_path, "rb") as f:
                ciphertext = f.read()
            sha_enc = sha256_file(enc_path)
            start = time.perf_counter()
            aes_decrypt_file(enc_path, work_path, key, iv)
            decrypt_time = (time.perf_counter() - start) * 1000

        # Integrity check: if the round-trip doesn't return the exact same
        # bytes, it's a real bug in the cipher implementation, not a
        # leftover-file problem. Better to fail loudly here than to
        # discover it later through a corrupted dataset.
        with open(work_path, "rb") as f:
            recovered = f.read()
        if recovered != data:
            raise RuntimeError(
                f"{cipher} round-trip did not match the original "
                f"for '{original_name}'. Check the cipher implementation."
            )

        freq_enc = byte_frequency_normalized(ciphertext)
        entropy_enc = shannonEntropy(ciphertext)

        # Bit-level (not byte-level) measurement: must happen here, while
        # enc_path still exists inside tmp_dir -- it's gone the moment we
        # leave this `with` block.
        bernoulli_p, bit_entropy_val = compute_bit_level_entropy(enc_path)

    # By the time we exit the `with`, tmp_dir (copy, .enc/.rc4/.aes) has
    # been fully removed, whether we succeeded or raised. data/ was never
    # touched.

    return {
        "run_id": run_id,
        "algorithm": cipher,

        "file_name": original_name,
        "size_bytes": len(data),

        "entropy_raw": entropy_raw,
        "entropy_encrypted": entropy_enc,
        "entropy_delta": entropy_enc - entropy_raw,

        "bernoulli_p": bernoulli_p,
        "bit_entropy": bit_entropy_val,

        "encrypt_time_ms": encrypt_time,
        "decrypt_time_ms": decrypt_time,

        "key": key,
        "iv": iv,

        "freq_raw": freq_raw,
        "freq_encrypted": freq_enc,

        "sha_raw": sha_raw,
        "sha_encrypted": sha_enc
    }


def run_cipher_algorithm(conf: dict):
    ds = Dataset(csv_name=conf.get("csvName", "global"))
    ds.init_files()
    files_ = get_files(conf["savedData"])
    if conf["algo"] == "All":
        algorithms = ["DES", "RC4", "AES"]
    else:
        algorithms = [conf["algo"]]

    max_workers = conf.get("workers") or os.cpu_count() or 1
    experiment_id = ds.next_experiment_id()

    for algorithm in algorithms:
        for loop_id in range(conf["loops"]):
            # Actual parallelism: ctypes releases the GIL while inside the
            # C calls (des_encrypt_file/decrypt_file), so these threads can
            # genuinely run on different cores at the same time. CSV writes
            # happen afterwards, sequentially, in the main thread only.
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(run_once, file_path=fp, cipher=algorithm, run_id=0)
                    for fp in files_
                ]
                results = [f.result() for f in futures]

            for result in results:
                ds.append_experiment([
                    experiment_id,
                    loop_id + 1,
                    experiment_id,
                    result["algorithm"],
                    result["file_name"],
                    result["size_bytes"],
                    result["entropy_raw"],
                    result["entropy_encrypted"],
                    result["entropy_delta"],
                    result["encrypt_time_ms"],
                    result["decrypt_time_ms"],
                    result["sha_raw"],
                    result["sha_encrypted"],
                    result["bernoulli_p"],
                    result["bit_entropy"]
                ])
                ds.append_key([
                    experiment_id,
                    result["algorithm"],
                    result["key"].hex(),
                    "" if result["iv"] is None else result["iv"].hex()
                ])
                experiment_id += 1


def run(conf):
    if not conf["onplots"]:
        run_cipher_algorithm(conf)

    csv_name = conf.get("csvName", "global")
    csv_dir = ROOT / "data" / "csv"

    if not conf.get("noPlots", False):
        csv_source = (
            str(csv_dir / "*.csv") if conf.get("combinePlots", False)
            else str(csv_dir / f"{csv_name}.csv")
        )
        plots = PlotGenerator(
            csv_file=csv_source,
            output_dir=conf["savedPlot"]
        )
        plots.generate_all()

    if conf.get("keyAnalysis", False):
        # Note the different glob: KeyStatistics wants the *_keys.csv
        # files specifically (PlotGenerator's glob explicitly EXCLUDES
        # them, since those hold cryptographic material, not metrics).
        keys_source = (
            str(csv_dir / "*_keys.csv") if conf.get("combinePlots", False)
            else str(csv_dir / f"{csv_name}_keys.csv")
        )
        key_stats = KeyStatistics(
            csv_file=keys_source,
            output_dir=conf["savedPlot"]
        )
        key_stats.generate_all()