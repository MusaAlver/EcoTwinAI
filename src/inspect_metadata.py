from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "data_description_table_3year_clean_data.xlsx"
)


def main() -> None:
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata dosyası bulunamadı: {METADATA_FILE}"
        )

    workbook = pd.ExcelFile(METADATA_FILE)

    print("Excel sayfaları:")
    for sheet_name in workbook.sheet_names:
        print(f"- {sheet_name}")

    print("\nSayfa ön izlemeleri:")

    for sheet_name in workbook.sheet_names:
        dataframe = pd.read_excel(
            METADATA_FILE,
            sheet_name=sheet_name,
            nrows=8,
        )

        print("\n" + "=" * 80)
        print(f"SAYFA: {sheet_name}")
        print(f"Sütunlar: {list(dataframe.columns)}")
        print(dataframe.to_string(index=False))


if __name__ == "__main__":
    main()
