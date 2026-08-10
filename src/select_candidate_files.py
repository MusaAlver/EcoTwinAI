from pathlib import Path
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "reports" / "metadata_keyword_hits.csv"
OUTPUT_FILE = PROJECT_ROOT / "reports" / "candidate_signals.csv"

IMPORTANT_WORDS = (
    "energy|power|electric|occupancy|occupant|people|person|camera|"
    "wifi|co2|carbon dioxide|temperature|humidity|lighting|light|"
    "hvac|weather|solar"
)

def main() -> None:
    data = pd.read_csv(INPUT_FILE)

    selected = data[
        data["content"].str.contains(
            IMPORTANT_WORDS,
            case=False,
            na=False,
            regex=True,
        )
    ].copy()

    selected["csv_files"] = selected["content"].apply(
        lambda text: ", ".join(
            dict.fromkeys(
                re.findall(r"[A-Za-z0-9_]+\.csv", str(text))
            )
        )
    )

    selected = selected[
        ["sheet", "excel_row", "keywords", "csv_files", "content"]
    ]

    selected.to_csv(OUTPUT_FILE, index=False)

    print(f"Seçilen metadata satırı: {len(selected)}")
    print(f"Kaydedilen dosya: {OUTPUT_FILE}")

    print("\nBulunan CSV dosyaları:\n")

    files = sorted(
        {
            file_name.strip()
            for value in selected["csv_files"]
            for file_name in str(value).split(",")
            if file_name.strip()
        }
    )

    for file_name in files:
        print(f"- {file_name}")

if __name__ == "__main__":
    main()
