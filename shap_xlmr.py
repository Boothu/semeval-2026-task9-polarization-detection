import argparse
import csv
import os
import random

import shap
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Read CSV into list of rows
def read_csv(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# Converts list of rows into dictionary keyed by id
def build_id_map(rows, key="id"):
    return {r[key]: r for r in rows}


def main():
    # Argument parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, help="eng or spa")
    ap.add_argument("--model_dir", required=True, help="models/xlmr_eng or models/xlmr_spa")
    ap.add_argument("--dev_text", required=True, help="path to dev text CSV (id,text)")
    ap.add_argument("--dev_gold", required=True, help="path to dev gold CSV (id,polarization)")
    ap.add_argument("--out_dir", default="shap", help="output folder")
    ap.add_argument("--k", type=int, default=5, help="examples per class type (TP/TN/FP/FN)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed) # Set random seed

    # Load dev and gold data
    dev_rows = read_csv(args.dev_text)
    gold_rows = read_csv(args.dev_gold)
    gold_map = build_id_map(gold_rows)

    # Load fine-tuned XLM-R model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Takes list of texts and returns probability of class 1 for each text
    # SHAP will call this repeatedly
    def predict_p1(texts):
        toks = tokenizer(
            list(texts),
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.no_grad():
            logits = model(**toks).logits
            probs = F.softmax(logits, dim=-1)[:, 1]
        return probs.detach().cpu().numpy() # Return numpy array so SHAP can use it

    # Build TP/TN/FP/FN pools
    tp, tn, fp, fn = [], [], [], []
    # For every dev example
    for r in dev_rows:
        # Lookup true label from gold file
        rid = r["id"]
        text = r["text"]
        if rid not in gold_map:
            continue
        y = int(str(gold_map[rid]["polarization"]).strip())

        # Run model to get probability of label 1
        p1 = float(predict_p1([text])[0])
        
        # If models confidence is above 50% predict 1
        pred = 1 if p1 >= 0.5 else 0

        # Put example into one of the 4 pools
        item = {"id": rid, "text": text, "gold": y, "pred": pred, "p1": p1}
        if pred == 1 and y == 1:
            tp.append(item)
        elif pred == 0 and y == 0:
            tn.append(item)
        elif pred == 1 and y == 0:
            fp.append(item)
        else:
            fn.append(item)

    # Takes 'k' examples from each category and builds a list from them
    def sample(items, k):
        if len(items) <= k:
            return items
        return random.sample(items, k)

    chosen = (
        [(x, "TP") for x in sample(tp, args.k)] +
        [(x, "TN") for x in sample(tn, args.k)] +
        [(x, "FP") for x in sample(fp, args.k)] +
        [(x, "FN") for x in sample(fn, args.k)]
    )

    # Make output directory
    out_dir = os.path.join(args.out_dir, f"xlmr_{args.lang}")
    os.makedirs(out_dir, exist_ok=True)

    # Save csv listing which examples were explained
    with open(os.path.join(out_dir, "explained_examples.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "gold", "pred", "p1"])
        for item, t in chosen:
            w.writerow([item["id"], t, item["gold"], item["pred"], f"{item['p1']:.6f}"])

    # Extract texts for SHAP
    texts = [item["text"] for item, _ in chosen]

    # Create SHAP explainer and compute explanations
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_p1, masker)
    shap_values = explainer(texts)

    # Save SHAP explanations as HTML
    for i, (item, t) in enumerate(chosen):
        fname = f"{i:02d}_{t}_{item['id']}.html"
        out_path = os.path.join(out_dir, fname)
        try:
            html = shap.plots.text(shap_values[i], display=False)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html)
        except TypeError:
            # Save a single combined HTML as fallback
            combined = os.path.join(out_dir, "all_explanations.html")
            html = shap.plots.text(shap_values, display=False)
            with open(combined, "w", encoding="utf-8") as f:
                f.write(html)
            break

    print(f"Saved SHAP outputs to: {out_dir}")
    print(f"TP={len(tp)} TN={len(tn)} FP={len(fp)} FN={len(fn)}")


if __name__ == "__main__":
    main()