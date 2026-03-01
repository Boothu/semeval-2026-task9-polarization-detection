import csv
import os

import torch
from codecarbon import EmissionsTracker
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import torch.nn.functional as F

def output_predictions(dev_path, out_path, model_dir, lang):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Choose device (prefer GPU) and set eval mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Open CSV
    with open(dev_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    # CodeCarbon: emissions results saving format
    run_tag = f"xlmr_infer_{lang}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    emissions_dir = os.path.join("emissions", "xlmr", lang)
    os.makedirs(emissions_dir, exist_ok=True)

    # Setting up emissions tracker for inference
    tracker = EmissionsTracker(
        project_name=run_tag,
        output_dir=emissions_dir,
        output_file="xlmr_predict_emissions.csv",
        log_level="warning",
    )
    tracker.start()
    t0 = time.time() # To track runtime

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    behaviour_path = out_path.replace(".csv", "_behaviour.csv")  # Second output file for behaviour analysis

    # Write predictions in submission format + behaviour log (confidence + margin from logits)
    with open(out_path, "w", newline="", encoding="utf-8") as f, \
         open(behaviour_path, "w", newline="", encoding="utf-8") as b:

        writer = csv.writer(f)
        writer.writerow(["id", "polarization"])

        bw = csv.writer(b)
        bw.writerow(["id", "pred", "p0", "p1", "confidence", "logit_gap", "mode"])

        # Loop through set and make predictions
        for r in rows:
            text = r["text"]

            # Tokenise the text
            inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Run model without tracking gradients for efficiency
            with torch.no_grad():
                logits = model(**inputs).logits[0]

                # Prediction
                pred = int(torch.argmax(logits).item())

                # Behaviour tracking from logits
                # Convert logits into probabilities to record model confidence and decisiveness per example
                probs = F.softmax(logits, dim=-1)
                p0 = float(probs[0].item())
                p1 = float(probs[1].item())
                conf = max(p0, p1)
                gap = float(torch.abs(logits[1] - logits[0]).item())
                mode = "logits"

            writer.writerow([r["id"], pred])
            bw.writerow([r["id"], pred, p0, p1, conf, gap, mode])
            
    # Stop emissions tracking and runtime and print result
    secs = time.time() - t0
    kg_co2 = tracker.stop()

    print(f"[CodeCarbon] Inference emissions ({lang}): {kg_co2:.6f} kgCO2e")
    print(f"[Runtime] {secs:.1f}s total, {secs/len(rows):.3f}s/example, {(secs/len(rows))*1000:.1f}s per 1000")
    
    print(f"{len(rows)} rows output to: {out_path}")
    print(f"Wrote behaviour log to: {behaviour_path}")

# English
output_predictions("data/test/eng.csv", "testruns/xlmr/pred_eng.csv", "models/xlmr_eng", "eng")

# Spanish
output_predictions("data/test/spa.csv", "testruns/xlmr/pred_spa.csv", "models/xlmr_spa", "spa")
