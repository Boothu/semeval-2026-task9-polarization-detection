import csv
import argparse
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# Reads CSV, builds dictionary then returns it
def load_labels(path):
    labels = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["id"]] = int(row["polarization"])
    return labels


def main(pred_path, gold_path):
    # Load predictions and gold labels
    preds = load_labels(pred_path)
    golds = load_labels(gold_path)

    # Store in two lists
    y_true = []
    y_pred = []

    for id_, gold_label in golds.items():
        if id_ not in preds:
            continue
        y_true.append(gold_label)
        y_pred.append(preds[id_])

    # Compute macro-F1 and accuracy with scikit-learn
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    
    # Build a 2x2 confusion matrix for labels 0 and 1
    # Rows = true label, columns = predicted label
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() # Unpack the 2x2 matrix

    # Print results
    print(f"Examples evaluated: {len(y_true)}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro-F1:  {macro_f1:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="path to prediction CSV (id,polarization)")
    ap.add_argument("--gold", required=True, help="path to gold CSV (id,text,polarization)")
    args = ap.parse_args()

    main(args.pred, args.gold)