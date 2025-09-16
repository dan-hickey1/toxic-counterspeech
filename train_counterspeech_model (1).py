import os
import math
import json
import click
import torch
import torch.nn as nn
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Any, Optional

from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    RobertaModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset, DatasetDict


# --------------------------- Model -------------------------------------------
class RobertaTanhHeadForSequenceClassification(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 3, dropout: float = 0.5):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels

        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)

        hidden_size = self.roberta.config.hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None, # ignored by RoBERTa
        labels=None,
    ):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        x = self.fc1(cls)
        x = self.act(x)
        x = self.drop(x)
        logits = self.out(x)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


# --------------------------- Helpers -----------------------------------------
def ensure_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found columns: {list(df.columns)}")

def tokenize_pair(tokenizer, parent_col="parent", reply_col="reply", max_length=512):
    def _fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch[parent_col],
            text_pair=batch[reply_col],
            truncation=True,
            padding=False,
            max_length= max_length,
        )
    return _fn


def softmax_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


# --------------------------- CLI ---------------------------------------------
@click.command()
@click.option("--train_csv", type=click.Path(exists=True, dir_okay=False), default="../../data/gpt_4o_labels.csv",
              help="Path to training CSV with columns: parent, reply, label")
@click.option("--test_csv", type=click.Path(exists=True, dir_okay=False), default="../../data/validation_set.csv",
              help="Path to test CSV with columns: parent, reply (label optional)")
@click.option("--output_dir", type=click.Path(file_okay=False), default="../data/model_checkpoints",
              show_default=True, help="Where to save checkpoints and outputs")
@click.option("--model_name", default="roberta-base", show_default=True,
              help="HF model name or path")
@click.option("--epochs", default=10, show_default=True, help="Num training epochs")
@click.option("--batch_size", default=16, show_default=True, help="Per-device batch size")
@click.option("--lr", default=1e-5, show_default=True, help="Learning rate")
@click.option("--dropout", default=0.5, show_default=True, help="Dropout for custom head")
@click.option("--seed", default=1111, show_default=True, help="Random seed")
@click.option("--val_size", default=0.2, show_default=True, help="Validation split proportion")
@click.option("--max_length", default=512, show_default=True, help="Max token length")
def main(train_csv, test_csv, output_dir, model_name, epochs, batch_size, lr,
         dropout, seed, val_size, max_length):
    
    os.makedirs(output_dir, exist_ok=True)

    output_fn = train_csv.split('/')[-1].split('.')[0]
    # ------------------ Load CSVs ------------------
    train_df = pd.read_csv(train_csv)
    ensure_cols(train_df, ["parent", "reply", "label"])
    train_df["label"] = train_df["label"].astype(int)

    test_df = pd.read_csv(test_csv)
    ensure_cols(test_df, ["parent", "reply"])
    has_test_labels = "label" in test_df.columns
    if has_test_labels:
        test_df["label"] = test_df["label"].astype(int)

    # ------------------ Split train/val ------------------
    tr_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df["label"],
        random_state=seed
    )

    # ------------------ HF datasets ------------------
    ds = DatasetDict({
        "train": Dataset.from_pandas(tr_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
    })

    # ------------------ Tokenizer & Collator ------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok_fn = tokenize_pair(tokenizer, "parent", "reply", max_length=max_length)

    keep_cols = ["label"] if "label" in ds["train"].column_names else []
    ds_tok = ds.map(tok_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ------------------ Model ------------------
    num_labels = 3
    model = RobertaTanhHeadForSequenceClassification(
        model_name=model_name,
        num_labels=num_labels,
        dropout=dropout
    )

    # ------------------ Trainer ------------------
    args = TrainingArguments(
        output_dir=f'{output_dir}/{output_fn}',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=100,
        save_total_limit=1,
        report_to="none",
        seed=seed,
        dataloader_num_workers=2,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # ------------------ Train ------------------
    trainer.train()

    # Save path to best checkpoint for reproducibility
    best_ckpt = trainer.state.best_model_checkpoint
    meta = {
        "best_model_checkpoint": best_ckpt,
        "train_csv": os.path.abspath(train_csv),
        "test_csv": os.path.abspath(test_csv),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "dropout": dropout,
        "seed": seed,
        "val_size": val_size,
        "model_name": model_name,
    }
    with open(os.path.join(f'{output_dir}/{output_fn}', "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ------------------ Predict on test ------------------
    # (Best model is already loaded due to load_best_model_at_end=True)
    preds = trainer.predict(ds_tok["test"])
    logits = torch.tensor(preds.predictions)
    probs = softmax_logits(logits).cpu().numpy()
    pred_labels = probs.argmax(axis=-1)

    # ------------------ Save predictions ------------------
    out = test_df.copy()
    out["pred_label"] = pred_labels
    out["prob_other"] = probs[:, 0]
    out["prob_hs"] = probs[:, 1]
    out["prob_cs"] = probs[:, 2]

    out_path = os.path.join(f"../../data/predictions/{output_fn}_predictions.csv")
    out.to_csv(out_path, index=False)

    print(f"Best checkpoint: {best_ckpt}")
    print(f"Predictions saved to: {out_path}")


if __name__ == "__main__":
    main()
