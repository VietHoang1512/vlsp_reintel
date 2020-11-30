import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

import random
import math

from tqdm import tqdm
import string
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import backend as K

import transformers
from transformers import TFAutoModel, TFRobertaModel, AutoTokenizer, TFAutoModelForSequenceClassification,TFAutoModelForSequenceClassification

from collections import Counter

print('Using Tensorflow version:', tf.__version__)
print('Using Transformers version:', transformers.__version__)

# import warnings
# warnings.filterwarnings('ignore')
tqdm.pandas()

def seed_all(seed=1512):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

train_df = pd.read_csv("/home/leonard/leonard/nlp/ReINTEL/data/final_data/train_5_folds.csv")
test_df = pd.read_csv("/home/leonard/leonard/nlp/ReINTEL/data/final_data/test.csv")

train_df["post_message"] = train_df["post_message"].astype(str)
test_df["post_message"] = test_df["post_message"].astype(str)

def tokenized_text_normalize(text):
#     text = text.strip(string.punctuation+" ")
    text = re.sub("\*", " ", text)
    text = re.sub("#", " ", text)
    text = text.strip(r"# *" )
    text = re.sub(" _ ",  " ", text)
    text = re.sub("\.+",  "\.", text)
    text = re.sub("…",  "\.", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"- - -", " ", text)
    text = re.sub("_ ", " ", text)
    text = re.sub(" +", " ", text)
    return text

train_df["post_message"] = train_df["post_message"].apply(tokenized_text_normalize)
test_df["post_message"] = test_df["post_message"].apply(tokenized_text_normalize)

def tokenized_text_normalize(text):
#     text = text.strip(string.punctuation+" ")
    text = re.sub("\*", " ", text)
    text = re.sub("#", " ", text)
    text = text.strip(r"# *" )
    text = re.sub(" _ ",  " ", text)
#     text = re.sub("\.+",  "\.", text)
    text = re.sub("…",  "...", text)
#     text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"- - -", " ", text)
    text = re.sub("_ ", " ", text)

    return text

train_df["post_message"] = train_df["post_message"].apply(tokenized_text_normalize)
test_df["post_message"] = test_df["post_message"].apply(tokenized_text_normalize)

nan_dict = {"timestamp_post": 1e10, "num_like_post":-2,"num_comment_post":-2, "num_share_post":-2}
unknown_dict = {"timestamp_post": 1e10, "num_like_post":-2,"num_comment_post":-2, "num_share_post":-2}

def extract_num(text):
    return re.findall("\d+", text)[0]
def vlsp_impute(text, field):
    if type(text)!=str:
        if math.isnan(text):
            return nan_dict[field]
    elif text=="unknown":        
        return unknown_dict[field]
    else:
        try:
            return int(extract_num(text))
        except:
            print(text)
            return nan_dict[field]

for field in tqdm([ "num_like_post", "num_comment_post", "num_share_post"]):
    # with open(f"train_{field}.txt", "w") as f:
    #     f.write("\n".join([str(x) for x in train_df[field].unique()]))
    # with open(f"test_{field}.txt", "w") as f:
        # f.write("\n".join([str(x) for x in test_df[field].unique()]))    
    train_df[field] =  train_df[field].apply(lambda text: vlsp_impute(text, field)).astype(int)
    test_df[field] =  test_df[field].apply(lambda text: vlsp_impute(text, field)).astype(int)

scaler = StandardScaler()
scaler.fit(pd.concat([train_df, test_df])[["timestamp_post", "num_like_post", "num_comment_post", "num_share_post"]].values)

train_df[["timestamp_post", "num_like_post", "num_comment_post", "num_share_post"]] = scaler.transform(train_df[["timestamp_post", "num_like_post", "num_comment_post", "num_share_post"]])
test_df[["timestamp_post", "num_like_post", "num_comment_post", "num_share_post"]] = scaler.transform(test_df[["timestamp_post", "num_like_post", "num_comment_post", "num_share_post"]])

train_df["timestamp_post"] = train_df["timestamp_post"].fillna(0)
test_df["timestamp_post"] = test_df["timestamp_post"].fillna(0)

roberta = '/home/leonard/leonard/nlp/ReINTEL/outputs/pretraining_vlsp' 
roberta_tokenizer = AutoTokenizer.from_pretrained(roberta)

MAX_LEN=256

def regular_encode(df, max_len=MAX_LEN):
    
    roberta_enc_di = roberta_tokenizer.batch_encode_plus(
        df["post_message"].values,
        return_token_type_ids=True,
        pad_to_max_length=True,
        max_length=max_len,
        truncation=True,
    )
    timestamp_post = (df["timestamp_post"]).values
    num_like_post = (df["num_like_post"]).values
    num_like_post = (df["num_like_post"]).values
    num_comment_post = (df["num_comment_post"]).values
    num_share_post = (df["num_share_post"]).values
    
    roberta_enc = (
        np.array(roberta_enc_di["input_ids"]),
        np.array(roberta_enc_di["attention_mask"]),
        np.array(roberta_enc_di["token_type_ids"]),
        timestamp_post,
        num_like_post,
        num_comment_post,
        num_share_post,
#         np.concatenate([num_like_post, num_comment_post, num_share_post], axis=0)
    )
    
    
    return roberta_enc

AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16

def data_generator(train_df, val_df):

    X_train = regular_encode(train_df, max_len=MAX_LEN)
    # y_train = tf.keras.utils.to_categorical(train_df['Label'].values, num_classes=2)
    y_train = train_df["label"].values
    X_val = regular_encode(val_df, max_len=MAX_LEN)
    # y_val = tf.keras.utils.to_categorical(val_df['Label'].values, num_classes=2)
    y_val = val_df["label"].values

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .repeat()
        .shuffle(1024)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    valid_dataset = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTO)
    )

    return train_dataset, valid_dataset

def build_model(bert_model_name_or_path="vinai/phobert-base", max_len=384, n_hiddens=-1):
    bert_model = TFAutoModel.from_pretrained(bert_model_name_or_path)

    bert_input_word_ids = tf.keras.layers.Input(
        shape=(max_len,), dtype=tf.int32, name="bert_input_id"
    )
    bert_attention_mask = tf.keras.layers.Input(
        shape=(max_len,), dtype=tf.int32, name="bert_attention_mask"
    )
    bert_token_type_ids = tf.keras.layers.Input(
        shape=(max_len,), dtype=tf.int32, name="bert_token_type_ids"
    )

    bert_sequence_output = bert_model(
        bert_input_word_ids,
        attention_mask=bert_attention_mask,
        token_type_ids=bert_token_type_ids,
        output_hidden_states=True,
        output_attentions=True,
        
    )

    # print(len(bert_sequence_output)) # 4

    # print(bert_sequence_output[0].shape) # (None, max_len, 768)

    # print(bert_sequence_output[1].shape) # (None, 768)
    # print(len(bert_sequence_output[2])) # 13
    # print(bert_sequence_output[2][0].shape) # (None, max_len, 768)
    # print(len(bert_sequence_output[3])) # 12
    # print(bert_sequence_output[3][0].shape) # (None, 12, None, max_len)

    # TODO: get bert embedding

    if n_hiddens == -1:  # get [CLS] token embedding only
        # print("Get pooler output of shape (batch_size, hidden_size)")
        bert_sequence_output = bert_sequence_output[0][:, 0, :]
#         bert_sequence_output = bert_sequence_output[1]
    else:  # concatenate n_hiddens final layer
        # print(f"Concatenate {n_hiddens} hidden_states of shape (batch_size, hidden_size)")
        bert_sequence_output = tf.concat(
            [bert_sequence_output[2][-i] for i in range(n_hiddens)], axis=-1)
        bert_sequence_output = bert_sequence_output[:, 0, :]

    # print("bert_sequence_output shape", bert_sequence_output.shape)

#     bert_output = tf.keras.layers.Flatten()(bert_sequence_output)
    bert_output = tf.keras.layers.Dense(8, activation="relu")(bert_sequence_output)
#     print(bert_output.shape)
    
    timestamp_post = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name="timestamp_post")
    num_like_post = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name="num_like_post")
    num_comment_post = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name="num_comment_post")
    num_share_post = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name="num_share_post")
    
    aulixiary_info = tf.keras.layers.Concatenate()([timestamp_post, num_like_post, num_comment_post, num_share_post])
#     aulixiary_output  = tf.keras.layers.GaussianNoise(0.2)(aulixiary_info)
    
    out = tf.keras.layers.Concatenate()([bert_output, aulixiary_info]) 
#     print(out.shape)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(out)

    model = tf.keras.models.Model(
        inputs=[
            bert_input_word_ids,
            bert_attention_mask,
            bert_token_type_ids,  # bert input
            timestamp_post,
            num_like_post,
            num_comment_post,
            num_share_post,
        ],
        outputs=out,
    )
    model.compile(
        tf.keras.optimizers.Adam(lr=5e-5),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC()],
    )

    return model

n_splits = 5
n_epochs = 5

DISPLAY=1 # USE display=1 FOR INTERACTIVE
exp = f'phobert+auxiliary_{MAX_LEN}_len'

output_dir = f'../outputs/{exp}_models'
os.makedirs(output_dir, exist_ok=True)

def scheduler(epoch):
    return 3e-5*0.2**epoch

# for fold, (idxT, idxV) in enumerate(kf.split(train_df)):
# for fold in sorted(train_df["fold"].unique()):
#     print('*'*100)
#     print(f'FOLD: {fold+1}/{n_splits}')
    
#     K.clear_session()
#     # with strategy.scope():
#     #     model = build_model(max_len=MAX_LEN)
#     model = build_model(max_len=MAX_LEN)

        
#     reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

#     model_dir = os.path.join(output_dir, f'Fold_{fold+1}.h5')

#     sv = tf.keras.callbacks.ModelCheckpoint(model_dir, 
#                                             monitor='val_auc', 
#                                             verbose=1, 
#                                             save_best_only=True,
#                                             save_weights_only=True, 
#                                             mode='max', 
#                                             save_freq='epoch')
    
#     train_df_ = train_df[train_df["fold"]!=fold]
#     val_df_ = train_df[train_df["fold"]==fold]
#     train_dataset, valid_dataset = data_generator(train_df_, val_df_)
    
#     n_steps = train_df_.shape[0] // BATCH_SIZE + 1
#     train_history = model.fit(
#         train_dataset,
#         steps_per_epoch=n_steps,
        
#         callbacks=[sv, 
#             reduce_lr,
#             # tb
#             ],
#         validation_data=valid_dataset,
#         epochs=n_epochs
#     )

X_test = regular_encode(test_df, max_len=MAX_LEN)
y_test = np.zeros((len(test_df),1))
test_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_test,y_test))
    .batch(BATCH_SIZE)
)

model = build_model(max_len=MAX_LEN)
preds = []
for i, file_name in enumerate(os.listdir(output_dir)):
    print('_'*80)
    
    K.clear_session()
    model_path = os.path.join(output_dir, file_name)
    
    print(f'Inferencing with model from: {model_path}')
    model.load_weights(model_path)

    pred = model.predict(test_dataset,
                         batch_size=128,
                         verbose=DISPLAY)
    # print(pred[])
    preds.append(pred)

preds = np.mean(preds, axis=0)
test_df["prediction"] = preds
test_df["prediction"].to_csv(f"{exp}.csv", header=False)