import argparse
import csv
import os
import random
import re
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from codecarbon import EmissionsTracker
from datetime import datetime
import time
import torch.nn.functional as F

# Create regex that finds a 0 or 1 anywhere in model output
LABEL_RE = re.compile(r"([01])")

# Read CSV into list of rows
def read_csv(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# Builds Llama prompt
def build_prompt(text: str, lang: str, fewshot_examples):
    # Explain task and output format (use Spanish prompt for Spanish dataset, English otherwise)
    if lang == "spa":
        header = (
            "Instrucciones: Detecta la polarización de actitudes en el siguiente texto.\n"
            "Si el texto muestra una fuerte comparación de 'nosotros contra ellos', desprecio, culpa o declaraciones negativas generalizadas realizadas por un grupo o un bando, el resultado tiene que ser 1. En caso contrario el resultado tiene que ser 0.\n"
            "Regla: Devolver SOLO un carácter: 0 o 1"
        )
        
        fs = ""
        
        # If fewshot examples exist add them to the prompt
        if fewshot_examples:
            fs += "\nEjemplos:\n"
            for i, (ex_text, ex_label) in enumerate(fewshot_examples, start=1):
                fs += f"{i}) Texto: {ex_text}\n   Respuesta: {ex_label}\n"

        query = f"\nAhora clasifica este texto en espanol:\nTexto: {text}\nRespuesta:"
        return header + fs + query
    
    else:
        header = (
            "Task: Attitude polarization detection.\n"
            "Output 1 if the text shows strong us-vs-them framing, contempt, blame, or generalised negative claims toward a group or side. Output 0 otherwise.\n"
            "Rule: Return ONLY a single character: 0 or 1.\n"
        )

        fs = ""
        
        if fewshot_examples:
            fs += "\nExamples:\n"
            for i, (ex_text, ex_label) in enumerate(fewshot_examples, start=1):
                fs += f"{i}) Text: {ex_text}\n   Answer: {ex_label}\n"

        query = f"\nNow classify this {lang} text:\nText: {text}\nAnswer:"
        return header + fs + query

# Parse Llama output into a label - looks for 0 or 1 
# Returns (-1, mode) if no digit is found (mode indicates why -1 was returned)
def parse_label(s: str):
    s = (s or "").strip()
    if not s:
        return -1, "empty"
    if s[0] in ("0", "1"):
        return int(s[0]), "first_char"
    m = LABEL_RE.search(s)
    if m:
        return int(m.group(1)), "fallback"
    return -1, "no_match"  

# Loads Llama model using Unsloth with 4-bit quantization reducing memory usage and improving efficiency
def load_llm(model_name: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,  # Quantized
    )
    FastLanguageModel.for_inference(model)
    model.eval()
    return tokenizer, model

# Generates up to 2 new tokens (aiming for 0/1) using greedy decoding
def hf_generate(tokenizer, model, prompt: str, seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Decode generated tokens
    out_ids = outputs.sequences
    new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Looks at the models first generated step and compares how strongly it prefers '0' against '1' by reading logits 
    # Converts logits into probabilities to record models confidence and decisiveness for each prediction, even if the final output text needs parsing
    try:
        step0_logits = outputs.scores[0][0]  # Logits for the first generated token

        # Helper to safely get token id
        def tok_id(tok: str):
            tid = tokenizer.convert_tokens_to_ids(tok)
            unk = getattr(tokenizer, "unk_token_id", None)
            if tid is None:
                return None
            if unk is not None and tid == unk:
                return None
            return tid

        # Find which token represents 0 and 1
        id0 = tok_id("▁0") or tok_id("0")
        id1 = tok_id("▁1") or tok_id("1")

        # If both token ids found, score them
        if id0 is not None and id1 is not None:
            logit0 = step0_logits[id0].item()
            logit1 = step0_logits[id1].item()

            # Convert to probabilities
            probs = F.softmax(torch.tensor([logit0, logit1]), dim=0)
            p0 = float(probs[0].item())
            p1 = float(probs[1].item())
            conf = max(p0, p1)
            gap = float(abs(logit1 - logit0))
            mode = "step0_logits"
        else:
            p0, p1, conf, gap, mode = None, None, None, None, "digit_token_not_found"
    except Exception:
        p0, p1, conf, gap, mode = None, None, None, None, "no_scores"

    return gen_text, p0, p1, conf, gap, mode

# Chooses 'k' fewshot examples from train set and returns as list
# Tries to pick a balanced set of examples (roughly half 0, half 1)
def pick_fewshot_examples(train_rows, k: int, seed: int):
    if k <= 0:
        return []

    rnd = random.Random(seed)
    zeros = [r for r in train_rows if str(r.get("polarization", "")).strip() == "0"]
    ones  = [r for r in train_rows if str(r.get("polarization", "")).strip() == "1"]

    k0 = k // 2
    k1 = k - k0

    ex0 = rnd.sample(zeros, k=min(k0, len(zeros))) if zeros else []
    ex1 = rnd.sample(ones,  k=min(k1, len(ones)))  if ones else []

    examples = [(r["text"], int(str(r["polarization"]).strip())) for r in (ex0 + ex1)]
    rnd.shuffle(examples)
    return examples

# Main loop - run on dev set and write output file
def run(dev_path: str, out_path: str, lang: str, model: str, seed: int,
        train_path: str = None, fewshot_k: int = 0):

    dev_rows = read_csv(dev_path)
    
    # CodeCarbon: emissions results saving format
    run_tag = f"llama_infer_{lang}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    emissions_dir = os.path.join("emissions", "llama", lang)
    os.makedirs(emissions_dir, exist_ok=True)

    # Setting up emissions tracker for inference
    tracker = EmissionsTracker(
        project_name=run_tag,
        output_dir=emissions_dir,
        output_file="llama_emissions.csv",
        log_level="warning",
    )
    tracker.start()

    fewshot_examples = []
    if fewshot_k > 0:
        if not train_path:
            raise ValueError("fewshot_k > 0 requires --train <path_to_train_csv>")
        train_rows = read_csv(train_path)
        fewshot_examples = pick_fewshot_examples(train_rows, fewshot_k, seed)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    tokenizer, llm = load_llm(model)
    
    t0 = time.time() # To track runtime
    
    behaviour_path = out_path.replace(".csv", "_behaviour.csv") # Second output file for behaviour analysis

    # Open behaviour file aswell as prediction output file
    with open(out_path, "w", newline="", encoding="utf-8") as f, \
        open(behaviour_path, "w", newline="", encoding="utf-8") as b:

        w = csv.writer(f)
        w.writerow(["id", "polarization"])

        bw = csv.writer(b)
        bw.writerow(["id", "pred", "p0", "p1", "confidence", "logit_gap", "mode"])
        
        # bad_count indicates outputs where no 0/1 was found
        # fallback_count indicates outputs that did not start with 0/1 but contained one later
        bad_count = 0
        fallback_count = 0
        
        for r in dev_rows:
            prompt = build_prompt(r["text"], lang, fewshot_examples)

            # Get models text output and confidence signals from logits
            resp, p0, p1, conf, gap, logit_mode = hf_generate(tokenizer, llm, prompt, seed)

            label, mode = parse_label(resp)
            if mode == "fallback":
                fallback_count += 1
            if label == -1:
                bad_count += 1
                label = 0 # Fallback to keep output valid
            w.writerow([r["id"], label])
            bw.writerow([r["id"], label, p0, p1, conf, gap, logit_mode])

    # Stop emissions tracking and runtime and print results
    secs = time.time() - t0
    kg_co2 = tracker.stop()

    print(f"[CodeCarbon] LLaMA inference emissions ({lang}): {kg_co2:.6f} kgCO2e")
    print(f"[Runtime] {secs:.1f}s total, {secs/len(dev_rows):.3f}s/example, {(secs/len(dev_rows))*1000:.1f}s per 1000")
    
    print("bad_count:", bad_count, "fallback_count:", fallback_count)
    print(f"Wrote {len(dev_rows)} rows to {out_path}")
    print(f"Wrote behaviour log to {behaviour_path}")

# Command line arguments
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--lang", required=True, help="eng or spa (only used in prompt text)")
    ap.add_argument("--dev", required=True, help="path to dev CSV (must have id,text)")
    ap.add_argument("--out", required=True, help="output prediction CSV")
    ap.add_argument("--train", default=None, help="path to train CSV (needed for few-shot)")
    ap.add_argument("--fewshot_k", type=int, default=0, help="0 = zero-shot, else use K examples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run(
        dev_path=args.dev,
        out_path=args.out,
        lang=args.lang,
        model=args.model,
        seed=args.seed,
        train_path=args.train,
        fewshot_k=args.fewshot_k,
    )

if __name__ == "__main__":
    main()
