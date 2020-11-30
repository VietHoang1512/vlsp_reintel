import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from data.tokenizer import VnCoreTokenizer

def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed = 1512    
seed_all(seed)


def kfold_split(train_df, n_folds):
    train_df["fold"] = 0
    for label in tqdm(train_df["label"].unique()):
        class_index = train_df[train_df["label"]==label].index.tolist()
        random.shuffle(class_index)
        fold_len = len(class_index)//n_folds
        for fold in range(n_folds):
            train_df.loc[class_index[fold*fold_len:(fold+1)*fold_len], 'fold'] = fold
    return train_df

if __name__=="__main__":
    vncore_tokenizer = VnCoreTokenizer()
    # n_folds = 5
    # warmup_train_df = pd.read_excel("/home/leonard/leonard/nlp/ReINTEL/data/raw_data/warmup_training_dataset.xlsx", index_col="id")
    # warmup_test_df = pd.read_excel("/home/leonard/leonard/nlp/ReINTEL/data/raw_data/warmup_test_set.xlsx", index_col="id")

    # public_train_df = pd.read_csv("/home/leonard/leonard/nlp/ReINTEL/data/raw_data/public_train.csv")
    # public_test_df = pd.read_csv("/home/leonard/leonard/nlp/ReINTEL/data/raw_data/public_test.csv")
    private_test_df = pd.read_csv("/home/leonard/leonard/nlp/ReINTEL/data/final_data/final_private_test_dropped_no_label - final_private_test_dropped_no_label.csv")
    # TODO: make use of warmup_test_df
    # train_df = pd.concat([warmup_train_df, public_train_df]).drop_duplicates()
    # train_df = kfold_split(train_df, n_folds=5)
    # train_df["post_message"] = train_df["post_message"].astype(str).apply(vncore_tokenizer.tokenize)
    # public_test_df["post_message"] = public_test_df["post_message"].astype(str).apply(vncore_tokenizer.tokenize)
    private_test_df["post_message"] = private_test_df["post_message"].astype(str).apply(vncore_tokenizer.tokenize)
    
    # train_df.to_csv(f'/home/leonard/leonard/nlp/ReINTEL/data/tokenized_data/train_{n_folds}_folds.csv', index=False)
    # public_test_df.to_csv(f'/home/leonard/leonard/nlp/ReINTEL/data/tokenized_data/test.csv')
    private_test_df.to_csv(f'/home/leonard/leonard/nlp/ReINTEL/data/final_data/private_test.csv')
