from pathlib import Path
import csv


class Dataset:

    columns_experiment = [
        "experiment_id",
        "loop_id",
        "run_id",
        "algorithm",
        "file_name",
        "size_bytes",
        "entropy_raw",
        "entropy_encrypted",
        "entropy_delta",
        "encrypt_time_ms",
        "decrypt_time_ms",
        "sha256_raw",
        "sha256_encrypted"
    ]

    columns_keys = [
        "experiment_id",
        "algorithm",
        "key",
        "iv"
    ]

    def __init__(self, csv_name="global"):
        script_dir = Path(__file__).resolve().parent
        self.base_dir = script_dir.parent.parent

        csv_dir = self.base_dir / "data/csv"
        self.global_csv = csv_dir / f"{csv_name}.csv"
        self.keys_csv = csv_dir / f"{csv_name}_keys.csv"

        self.global_csv.parent.mkdir(parents=True, exist_ok=True)

    def init_files(self):

        if not self.global_csv.exists():
            with open(self.global_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.columns_experiment)

        if not self.keys_csv.exists():
            with open(self.keys_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.columns_keys)

    def next_experiment_id(self):
        """
        Returns the next free experiment_id, continuing from whatever is
        already in global.csv instead of always restarting at 1. This
        keeps IDs unique and aligned between global.csv and keys.csv
        across separate job runs (e.g. one job per algorithm).
        """
        if not self.global_csv.exists():
            return 1

        max_id = 0
        with open(self.global_csv, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if not row:
                    continue
                try:
                    max_id = max(max_id, int(row[0]))
                except ValueError:
                    continue
        return max_id + 1

    def append_experiment(self, row):

        with open(self.global_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def append_key(self, row):

        with open(self.keys_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)