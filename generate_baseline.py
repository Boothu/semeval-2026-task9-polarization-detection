import csv
import os

def output_predictions(dev_path, out_path, default_label):
    # Open CSV and use DictReader so each row can be accessed then put rows into list
    with open(dev_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Create output CSV in correct submission format
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "polarization"])
        # For every row get id and label it '0'
        for r in rows:
            writer.writerow([r["id"], default_label])

    print(f"{len(rows)} rows output to: {out_path}")

output_predictions("data/dev/eng.csv", "baseline/pred_eng.csv", default_label=0)
output_predictions("data/dev/spa.csv", "baseline/pred_spa.csv", default_label=0)
