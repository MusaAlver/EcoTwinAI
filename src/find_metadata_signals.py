from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "data_description_table_3year_clean_data.xlsx"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "metadata_keyword_hits.csv"
)

KEYWORDS = [
    "energy",
    "power",
    "electric",
    "occupancy",
    "occupant",
    "people",
    "person",
    "camera",
    "wifi",
    "wi-fi",
    "co2",
    "carbon dioxide",
    "temperature",
    "humidity",
    "lighting",
    "light",
    "hvac",
    "weather",
    "solar",
]


def main() -> None:
    workbook = pd.ExcelFile(METADATA_FILE)
    matches = []

    for sheet_name in workbook.sheet_names:
        dataframe = pd.read_excel(
            METADATA_FILE,
            sheet_name=sheet_name,
            header=None,
        )

        for row_index, row in dataframe.iterrows():
            row_text = " | ".join(
                str(value)
                for value in row.tolist()
                if pd.notna(value)
            )

            row_text_lower = row_text.lower()

            found_keywords = [
                keyword
                for keyword in KEYWORDS
                if keyword in row_text_lower
            ]

            if found_keywords:
                matches.append(
                    {
                        "sheet": sheet_name,
                        "excel_row": row_index + 1,
                        "keywords": ", ".join(found_keywords),
                        "content": row_text,
                    }
                )

    result = pd.DataFrame(matches)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print(f"Bulunan ilgili satır: {len(result)}")
    print(f"Dosya oluşturuldu: {OUTPUT_FILE}")

    print("\nİlk sonuçlar:\n")
    print(result.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
