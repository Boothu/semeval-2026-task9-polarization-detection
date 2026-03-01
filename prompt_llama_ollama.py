import argparse
import csv
import os
import random
import re
import requests

# Set Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

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
            "Tarea: Detección de polarización de actitudes.\n"
            "Resultado 1 si el texto muestra una fuerte confrontación de 'nosotros contra ellos', desprecio, culpar o afirmaciones negativas generalizadas hacia un grupo o bando. Resultado 0 en caso contrario.\n"
            "Regla: Devolver SOLO un carácter: 0 o 1.\n"
        )
        
        fs = ""
        
        # If fewshot examples exist add them to the prompt
        if fewshot_examples:
            fs += "\nEjemplos:\n"
            for i, (ex_text, ex_label) in enumerate(fewshot_examples, start=1):
                fs += f"{i}) Texto: {ex_text}\n   Respuesta: {ex_label}\n"

        query = f"\nAhora clasifica este texto en espanol:\nTexto: {text}\nRespuesta: "
        return header + fs + query
    

    else:
        header = (
            "Task: Attitude polarization detection.\n"
            "Output 1 if the text shows strong 'us vs them' confrontation, contempt, blame, or broad negative generalisations about a group/side.\n"
            "Output 0 otherwise.\n"
            "Rule: Return ONLY a single character: 0 or 1.\n"
        )

        fs = ""
        if fewshot_examples:
            fs += "\nExamples:\n"
            for i, (ex_text, ex_label) in enumerate(fewshot_examples, start=1):
                fs += f"{i}) Text: {ex_text}\nLabel: {ex_label}\n"

        query = f"\nNow classify this text in English:\nText: {text}\nLabel: "
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

# Call Ollama and get response text
# Uses temperature 0 and short generation to encourage model to output solely 0 or 1
def ollama_generate(model: str, prompt: str, seed: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "raw": True, # Send prompt as is
        "options": {
            "temperature": 0.0,
            "seed": seed,
            "num_predict": 2,
            "stop": ["\n", " ", "\t", "\r", ".", ","], # Stops at whitespace/punctuation to prevent extra text after the label
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()

# Chooses 'k' fewshot examples from train set and returns as list
# Tries to pick a balanced set of examples (roughly half 0, half 1)
# Deterministic given seed: same K examples will be chosen each run for reproducibility
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

    fewshot_examples = []
    if fewshot_k > 0:
        if not train_path:
            raise ValueError("fewshot_k > 0 requires --train <path_to_train_csv>")
        train_rows = read_csv(train_path)
        fewshot_examples = pick_fewshot_examples(train_rows, fewshot_k, seed)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "polarization"])
        
        # bad_count indicates outputs where no 0/1 was found
        # fallback_count indicates outputs that did not start with 0/1 but contained one later
        bad_count = 0
        fallback_count = 0

        for r in dev_rows:
            prompt = build_prompt(r["text"], lang, fewshot_examples)
            # Use fixed seed for reproducibility
            resp = ollama_generate(model, prompt, seed)
            label, mode = parse_label(resp)
            if mode == "fallback":
                fallback_count += 1
            if label == -1:
                bad_count += 1
            w.writerow([r["id"], label])

    print("bad_count:", bad_count, "fallback_count:", fallback_count)
    print(f"Wrote {len(dev_rows)} rows to {out_path}")

# Command line arguments
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama2:7b")
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
