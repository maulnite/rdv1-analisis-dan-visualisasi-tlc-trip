from pathlib import Path
from typing import Iterable

import requests


DATA_DIR = Path("/opt/airflow/data")
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"

TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
TAXI_ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

DEFAULT_TAXI_TYPE = "yellow"
DEFAULT_MONTHS = ["2025-01"]


def ensure_directories() -> None:
    """
    Membuat folder data yang dibutuhkan pipeline.
    Folder ini berada di dalam container, tapi karena sudah di-mount,
    hasilnya akan muncul juga di D:\\rdv-tlc-project\\data.
    """
    for folder in [RAW_DIR, EXTERNAL_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    print("[INFO] Data directories are ready.")
    print(f"[INFO] RAW_DIR      : {RAW_DIR}")
    print(f"[INFO] EXTERNAL_DIR : {EXTERNAL_DIR}")


def download_file(url: str, output_path: Path) -> Path:
    """
    Download file dari URL ke output_path.
    Kalau file sudah ada dan ukurannya lebih dari 0 byte, download akan dilewati.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[SKIP] File already exists: {output_path}")
        return output_path

    print(f"[DOWNLOAD] {url}")
    print(f"[SAVE TO]  {output_path}")

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with output_path.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[DONE] Downloaded {output_path.name} ({size_mb:.2f} MB)")

    return output_path


def download_tlc_data(
    months: Iterable[str] = DEFAULT_MONTHS,
    taxi_type: str = DEFAULT_TAXI_TYPE,
) -> list[str]:
    """
    Download NYC TLC Trip Record Dataset berdasarkan bulan.
    Contoh file:
    yellow_tripdata_2025-01.parquet
    """
    ensure_directories()

    downloaded_files = []

    for month in months:
        filename = f"{taxi_type}_tripdata_{month}.parquet"
        url = f"{TLC_BASE_URL}/{filename}"
        output_path = RAW_DIR / filename

        file_path = download_file(url, output_path)
        downloaded_files.append(str(file_path))

    print("[RESULT] TLC files:")
    for file_path in downloaded_files:
        print(f"  - {file_path}")

    return downloaded_files


def download_taxi_zone_lookup() -> str:
    """
    Download taxi zone lookup.
    File ini dipakai untuk menerjemahkan LocationID menjadi Borough dan Zone.
    """
    ensure_directories()

    output_path = EXTERNAL_DIR / "taxi_zone_lookup.csv"
    file_path = download_file(TAXI_ZONE_LOOKUP_URL, output_path)

    print(f"[RESULT] Taxi zone lookup: {file_path}")
    return str(file_path)


def validate_ingestion_outputs(
    months: Iterable[str] = DEFAULT_MONTHS,
    taxi_type: str = DEFAULT_TAXI_TYPE,
) -> None:
    """
    Validasi sederhana untuk memastikan semua file hasil ingestion sudah ada.
    """
    expected_files = []

    for month in months:
        expected_files.append(RAW_DIR / f"{taxi_type}_tripdata_{month}.parquet")

    expected_files.append(EXTERNAL_DIR / "taxi_zone_lookup.csv")

    missing_files = []

    for file_path in expected_files:
        if not file_path.exists() or file_path.stat().st_size == 0:
            missing_files.append(str(file_path))

    if missing_files:
        raise FileNotFoundError(f"Missing ingestion output files: {missing_files}")

    print("[CHECK] All ingestion output files exist:")
    for file_path in expected_files:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  - {file_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    download_tlc_data()
    download_taxi_zone_lookup()
    validate_ingestion_outputs()