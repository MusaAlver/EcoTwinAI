from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "building59"
OUTPUT_FILE = PROJECT_ROOT / "reports" / "core_files_profile.txt"

CORE_FILES = [
    "elec.csv",
    "occ.csv",
    "wifi.csv",
    "zone_co2.csv",
    "zone_temp_interior.csv",
    "site_weather.csv",
    "ashp_meter.csv",
]


def locate_file(file_name: str) -> Path | None:
    matches = list(DATA_ROOT.rglob(file_name))

    if not matches:
        return None

    return matches[0]


def main() -> None:
    output_lines = []

    for file_name in CORE_FILES:
        output_lines.append("=" * 100)
        output_lines.append(f"DOSYA: {file_name}")

        file_path = locate_file(file_name)

        if file_path is None:
            output_lines.append("DURUM: DOSYA BULUNAMADI")
            output_lines.append("")
            continue

        try:
            sample = pd.read_csv(file_path, nrows=5)

            output_lines.append(f"YOL: {file_path}")
            output_lines.append(
                f"BOYUT: {file_path.stat().st_size / 1024**2:.2f} MB"
            )
            output_lines.append(
                f"SÜTUN SAYISI: {len(sample.columns)}"
            )
            output_lines.append(
                "SÜTUNLAR:"
            )

            for column in sample.columns:
                output_lines.append(f"  - {column}")

            output_lines.append("")
            output_lines.append("İLK 5 KAYIT:")
            output_lines.append(sample.to_string(index=False))
            output_lines.append("")

        except Exception as error:
            output_lines.append(f"HATA: {error}")
            output_lines.append("")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        "\n".join(output_lines),
        encoding="utf-8",
    )

    print(f"Profil oluşturuldu: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
