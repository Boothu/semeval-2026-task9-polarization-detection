import csv

def inspect_csv(path, has_label):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\nFile: {path}")
    print(f"Number of rows: {len(rows)}")
    print("Columns:", reader.fieldnames)

    if has_label:
        labels = {}
        for r in rows:
            label = r["polarization"]
            labels[label] = labels.get(label, 0) + 1
        print("Label distribution:", labels)

inspect_csv("data/train/eng.csv", has_label=True)
inspect_csv("data/train/spa.csv", has_label=True)

inspect_csv("data/dev/eng.csv", has_label=False)
inspect_csv("data/dev/spa.csv", has_label=False)