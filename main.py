import warnings

warnings.filterwarnings("ignore")

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

import transformers
from data.datasets import data_generator, regular_encode
from data.text import tokenized_text_normalize, vlsp_impute
from data.utils import seed_all
from trainer.model import build_model, scheduler
from transformers import AutoTokenizer

print("Using Tensorflow version:", tf.__version__)
print("Using Transformers version:", transformers.__version__)

parser = argparse.ArgumentParser()


parser.add_argument(
    "--train_file",
    type=str,
    default="../data/final_data/train_5_folds.csv",
    help="path to k-fold-splited and tokenized train data",
)

parser.add_argument(
    "--test_file",
    default="../data/final_data/private_test.csv",
    type=str,
    help="path to tokenized test data",
)

parser.add_argument(
    "--do_train",
    default=True,
    type=bool,
    help="whether train the pretrained model with provided train data",
)

parser.add_argument(
    "--do_infer",
    default=True,
    type=bool,
    help="whether predict the pretrained model with provided test data",
)

parser.add_argument(
    "--pretrained_bert",
    default="vinai/phobert-base",
    type=str,
    help="path to pretrained bert model path or directory",
)

parser.add_argument(
    "--max_len",
    default=256,
    type=int,
    help="max sequence length for padding and truncation",
)

parser.add_argument(
    "--batch_size",
    default=24,
    type=int,
    help="num examples per batch",
)

parser.add_argument(
    "--n_epochs",
    default=5,
    type=int,
    help="num epochs required for training",
)

parser.add_argument(
    "--output_dir",
    default="../outputs",
    type=str,
    help="path to output model weights directory",
)

parser.add_argument(
    "--seed",
    default=1710,
    type=int,
    help="seed for reproceduce",
)

args = parser.parse_args()

if __name__ == "__main__":
    #
    seed_all(args.seed)

    # Load and process data
    train_df = pd.read_csv(args.train_file)
    test_df = pd.read_csv(args.test_file)

    train_df["post_message"] = train_df["post_message"].astype(str)
    test_df["post_message"] = test_df["post_message"].astype(str)
    train_df["post_message"] = train_df["post_message"].apply(tokenized_text_normalize)
    test_df["post_message"] = test_df["post_message"].apply(tokenized_text_normalize)

    train_df = vlsp_impute(train_df)
    test_df = vlsp_impute(test_df)

    roberta = args.pretrained_bert
    roberta_tokenizer = AutoTokenizer.from_pretrained(roberta)
    model = build_model(roberta, max_len=args.max_len)
    model.summary()

    DISPLAY = 1  # USE display=1 FOR INTERACTIVE
    os.makedirs(args.output_dir, exist_ok=True)
    strategy = tf.distribute.MirroredStrategy()
    if args.do_train:
        folds = train_df["fold"].unique()
        for fold in sorted(folds):
            print("*" * 100)
            print(f"FOLD: {fold+1}/{len(folds)}")
            if fold < 4:
                continue
            K.clear_session()
            with strategy.scope():
                model = build_model(roberta, max_len=args.max_len)

            reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

            model_dir = os.path.join(args.output_dir, f"Fold_{fold+1}.h5")

            sv = tf.keras.callbacks.ModelCheckpoint(
                model_dir,
                monitor="val_auc",
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode="max",
                save_freq="epoch",
            )

            train_df_ = train_df[train_df["fold"] != fold]
            val_df_ = train_df[train_df["fold"] == fold]
            train_dataset, valid_dataset = data_generator(
                train_df_, val_df_, roberta_tokenizer, max_len=args.max_len, batch_size=args.batch_size
            )

            n_steps = train_df_.shape[0] // args.batch_size + 1
            train_history = model.fit(
                train_dataset,
                steps_per_epoch=n_steps,
                callbacks=[
                    sv,
                    reduce_lr,
                ],
                validation_data=valid_dataset,
                epochs=args.n_epochs,
            )
    if args.do_infer:
        X_test = regular_encode(test_df, roberta_tokenizer, max_len=args.max_len)
        y_test = np.zeros((len(test_df), 1))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(args.batch_size)
        model = build_model(roberta, max_len=args.max_len)
        preds = []
        for i, file_name in enumerate(os.listdir(args.output_dir)):
            print("*" * 100)

            K.clear_session()
            model_path = os.path.join(args.output_dir, file_name)

            print(f"Inferencing with model from: {model_path}")
            model.load_weights(model_path)

            pred = model.predict(test_dataset, batch_size=128, verbose=DISPLAY)
            preds.append(pred)

        preds = np.mean(preds, axis=0)
        test_df["prediction"] = preds
        test_df["prediction"].to_csv("submission.csv", header=False)
