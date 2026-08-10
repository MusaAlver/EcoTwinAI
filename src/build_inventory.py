from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "building59"
OUTPUT_FILE = PROJECT_ROOT / "reports" / "dataset_inventory.csv"


def inspect_csv(csv_path: Path) -> dict:
    """CSV dosyasının tamamını yüklemeden temel özelliklerini inceler."""

    try:
        sample = pd.read_csv(csv_path, nrows=5)

        return {
            "file_name": csv_path.name,
            "relative_path": str(csv_path.relative_to(PROJECT_ROOT)),
            "size_mb": round(csv_path.stat().st_size / 1024**2, 2),
            "column_count": len(sample.columns),
            "columns": " | ".join(map(str, sample.columns)),
            "first_timestamp": str(sample.iloc[0, 0]) if not sample.empty else "",
            "status": "OK",
        }

    except Exception as error:
        return {
            "file_name": csv_path.name,
            "relative_path": str(csv_path.relative_to(PROJECT_ROOT)),
            "size_mb": round(csv_path.stat().st_size / 1024**2, 2),
            "column_count": 0,
            "columns": "",
            "first_timestamp": "",
            "status": f"ERROR: {error}",
        }


def main() -> None:
    csv_files = sorted(DATA_ROOT.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"CSV bulunamadı: {DATA_ROOT}")

    records = [inspect_csv(csv_path) for csv_path in csv_files]
    inventory = pd.DataFrame(records)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUTPUT_FILE, index=False)

    print(f"Toplam CSV: {len(inventory)}")
    print(f"Başarılı okunan: {(inventory['status'] == 'OK').sum()}")
    print(f"Envanter kaydedildi: {OUTPUT_FILE}")

    print("\nDosyalar:")
    print(inventory[["file_name", "size_mb", "column_count", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()