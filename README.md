# Polarisation Detection (SemEval Subtask 1)

This repo contains code to train an XLM-RoBERTa-base model & prompt Llama 2 (7B) and fulfil the SemEval 2026 polarisation detection task (Subtask 1).

Dataset not included. Follow POLAR instructions to obtain it.

## Data

Datasets are stored in the data/ folder, split into train/ and dev/ sets.

CSV columns:
- id
- text
- polarization

## Setup XLM-RoBERTa

Create a virtual environment and install dependencies:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install PyTorch with GPU support (NVIDIA):

```
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Check GPU is being used for training:

```
python check_gpu.py
```

## Train XLM-RoBERTa

```
python train_xlmr.py
```

Trains xlm-roberta-base on data/train/{lang}.csv and saves the model(s) to:

models/xlmr_{lang}/

Training logs metrics to Weights & Biases. Run this once before training to enable experiment tracking:

```
wandb login
```

## Generate XLM-RoBERTa predictions

Generate predictions on the development set using a trained model:

```
python predict_xlmr.py
```

This loads trained models, runs predictions on dev dataset, then outputs to:
subtask_1/pred_{lang}.csv

## SHAP XLM-RoBERTa explainability script

Generate SHAP explanations for the fine-tuned XLM-R model. The script samples a small set of examples from each category (TP/TN/FP/FN) and saves explanations as HTML files.

Run SHAP:

```
python shap_xlmr.py --lang {lang} --model_dir models/xlmr_{lang} --dev_text data/dev/{lang}.csv --dev_gold data/dev_gold/{lang}.csv
```

Outputs are saved to:

shap/xlmr_eng/

## Setup Llama

Create a virtual environment and install dependencies:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Login to Hugging Face:

```
hf auth login
```

## Prompt Llama (Unsloth)

Zero-shot prompting on the development set (Must use Linux, WSL or Colab):

```
python prompt_llama_unsloth.py --lang {lang} --dev data\dev\{lang}.csv --out runs\llama\zero\pred_{lang}.csv
```

Few-shot prompting (example: 4-shot, using training data):

```
python prompt_llama_unsloth.py --lang {lang} --dev data\dev\{lang}.csv --train data\train\{lang}.csv --fewshot_k 4 --out runs\llama\fewshot4\pred_{lang}.csv
```

Prompts llama2:7b via unsloth on data/dev/{lang}.csv and saves the model's predictions to:

runs/llama/{type}/pred_{lang}.csv

## Evaluate Predictions

eval_predictions will evaluate a models predictions against the goldlabels and output the achieved accuracy and macro-F1:

```
python eval_predictions.py --pred ...\pred_{lang}.csv --gold data\dev_gold\{lang}.csv

```
