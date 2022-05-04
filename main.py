import argparse
import os
import pathlib


import pandas as pd
import numpy as np
# from pandas.core.internals.managers import _tuples_to_blocks_no_consolidate


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset

import datasets, transformers
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import wandb

def get_parser(
    parser=argparse.ArgumentParser(
        description="Train a model on the train the dataset for the patent competition.")
):
    parser.add_argument(
        "--train_file",
        type=pathlib.Path,
        help="path to the train file",
        default=pathlib.Path("./data/train.csv"),
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        help="device to run the experiment",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='distilbert-base-uncased',
        help="the model to use",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        default=pathlib.Path("models") / "distilbert-base-uncased",
        help="where to save models",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="distilbert-base-uncased",
        help="the name for the this run",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="the number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="the batch size to use",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="the learning rate to use",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="the weight decay to use",
    )
    return parser

##helper functions
class TrainDataset(Dataset):
    def __init__(self, df,tokenizer):
        self.inputs = df['input'].values.astype(str)
        self.targets = df['target'].values.astype(str)
        self.label = df['score'].values
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        inputs = self.inputs[item]
        targets = self.targets[item]
        label = self.label[item]

        return {
            **self.tokenizer(inputs, targets),
            'label': label.astype(np.float32)
        }

def compute_metric(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.reshape(len(predictions))
    return {
        'pearson': np.corrcoef(predictions, labels)[0][1]
    }

def train(args):
    # set up wandb
    wandb.init(project="patent-competition")

    # set up the model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)
    model.to(args.device)

    # set up the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)


    # set up the datasets
    train_df = pd.read_csv(args.train_file)
    train_df, va_df = train_test_split(train_df, test_size=0.2, random_state=42)
    tr_dataset = TrainDataset(train_df,tokenizer)
    va_dataset = TrainDataset(va_df,tokenizer)

    
    # set up the training arguments
    train_args = TrainingArguments(
        output_dir=args.save_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        metric_for_best_model="pearson",
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=args.run_name,
    )

    # trainer
    trainer = Trainer(
        model,
        train_args,
        train_dataset=tr_dataset,
        eval_dataset=va_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metric,
    )
    trainer.train()
    wandb.finish()
    model.save_pretrained(args.save_dir/"final")

def main(args):
    print("starting training")
    train(args)

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)