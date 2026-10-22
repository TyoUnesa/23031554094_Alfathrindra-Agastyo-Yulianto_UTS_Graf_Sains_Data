import os
import sys

import pandas as pd

SOURCE = os.path.join("data", "raw", "events.csv")
OUTPUT_DIR = os.path.join("data", "processed")

TRAIN_OUTPUT = os.path.join(OUTPUT_DIR, "events_training.csv")
TEST_OUTPUT = os.path.join(OUTPUT_DIR, "events_testing.csv")

CHUNK_SIZE = 100_000

TRAIN_START = pd.Timestamp("2015-05-17", tz="UTC")
TRAIN_END = pd.Timestamp("2015-07-18", tz="UTC")

TEST_START = pd.Timestamp("2015-07-26", tz="UTC")
TEST_END = pd.Timestamp("2015-08-11", tz="UTC")


def main():
    if not os.path.isfile(SOURCE):
        print(f"Error: File '{SOURCE}' tidak ditemukan!")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    columns = pd.read_csv(SOURCE, nrows=0).columns

    for output in (TRAIN_OUTPUT, TEST_OUTPUT):
        pd.DataFrame(columns=columns).to_csv(output, index=False)

    training_count = 0
    testing_count = 0
    invalid_timestamps = 0

    print(f"Membaca dataset: {SOURCE}")
    print("Memulai filter timestamp...")

    for chunk in pd.read_csv(SOURCE, chunksize=CHUNK_SIZE):
        timestamp_numeric = pd.to_numeric(
            chunk["timestamp"], errors="coerce"
        )

        dates = pd.to_datetime(
            timestamp_numeric,
            unit="ms",
            utc=True,
            errors="coerce"
        )

        invalid_timestamps += int(dates.isna().sum())

        train_mask = (
            (dates >= TRAIN_START) &
            (dates < TRAIN_END)
        )

        test_mask = (
            (dates >= TEST_START) &
            (dates < TEST_END)
        )

        training = chunk.loc[train_mask]
        testing = chunk.loc[test_mask]

        if not training.empty:
            training.to_csv(
                TRAIN_OUTPUT,
                mode="a",
                header=False,
                index=False
            )

        if not testing.empty:
            testing.to_csv(
                TEST_OUTPUT,
                mode="a",
                header=False,
                index=False
            )

        training_count += len(training)
        testing_count += len(testing)

    print("\nFilter selesai.")
    print(f"Jumlah baris training : {training_count:,}")
    print(f"Jumlah baris testing  : {testing_count:,}")
    print(f"Timestamp tidak valid : {invalid_timestamps:,}")
    print(f"File training: {TRAIN_OUTPUT}")
    print(f"File testing : {TEST_OUTPUT}")


if __name__ == "__main__":
    main()