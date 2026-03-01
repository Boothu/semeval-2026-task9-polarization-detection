import os

import evaluate
from codecarbon import EmissionsTracker
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import time

MODEL_NAME = "xlm-roberta-base"

# Trains model on given language
def train_for_language(lang):
    # Load training CSV
    train_path = f"data/train/{lang}.csv"
    ds = load_dataset("csv", data_files={"train": train_path})["train"]
    
    # Rename 'polarization' column so it can be used by Trainer and convert labels to integers
    ds = ds.rename_column("polarization", "labels")
    def cast_labels(batch):
        batch["labels"] = int(str(batch["labels"]).strip())
        return batch
    ds = ds.map(cast_labels)

    # Make validation split (20% of train data)
    split = ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]

    # Tokeniser: convert text into numbers the model can learn from
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    # Load XLM-R for classification (binary: 0 for not polarised, 1 for polarised)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Evaluation metric: macro-F1
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {"macro_f1": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]}

    # Create output directory
    out_dir = f"models/xlmr_{lang}"
    os.makedirs(out_dir, exist_ok=True)

    # Training settings
    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        report_to=["wandb"],
        run_name=f"xlmr-{lang}-lr1e5-ep5",
        seed=42,
    )

    # Adds padding so all texts in a batch are the same length, which is required to process them together
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # CodeCarbon: emissions results saving format
    run_tag = f"xlmr_train_{lang}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    emissions_dir = os.path.join("emissions", "xlmr", lang)
    os.makedirs(emissions_dir, exist_ok=True)

    # Setting up emissions tracker for training
    tracker = EmissionsTracker(
        project_name=run_tag,
        output_dir=emissions_dir,
        output_file="xlmr_train_emissions.csv",
        log_level="warning",
    )
    tracker.start()
    t0 = time.time() # To track runtime

    # Train
    trainer.train()
    
    # Stop emissions tracking and runtime and print result
    secs = time.time() - t0
    kg_co2 = tracker.stop()

    print(f"[CodeCarbon] Training emissions ({lang}): {kg_co2:.6f} kgCO2e")
    print(f"[Runtime] Training time: {secs:.1f}s total")
    
    # Print best score
    best_f1 = trainer.state.best_metric
    print(f"Best macro-F1 for {lang}: {best_f1:.4f}")

    # Save best model
    trainer.save_model(out_dir)
    print(f"Saved model to: {out_dir}")


if __name__ == "__main__":
    for lang in ["eng", "spa"]:
        train_for_language(lang)
