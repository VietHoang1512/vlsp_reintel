#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


import math
import os
import random
import re

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import transformers
from tensorflow.keras import backend as K
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, TFAutoModel

print("Using Tensorflow version:", tf.__version__)
print("Using Transformers version:", transformers.__version__)

# import warnings
# warnings.filterwarnings('ignore')
tqdm.pandas()


# In[3]:


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[4]:


seed = 1512


def seed_all(seed=1512):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)


seed_all(seed)


# In[5]:


# # %%time

# train_df = pd.read_csv("/home/leonard/leonard/nlp/ReINTEL/data/final_data/train_5_folds.csv")
# # test_df = pd.read_csv("/home/leonard/leonard/nlp/ReINTEL/data/final_data/test.csv")
# test_df = pd.read_csv("/home/leonard/leonard/nlp/ReINTEL/data/final_data/private_test.csv", index_col=0)


# train_df["post_message"] = train_df["post_message"].astype(str)
# test_df["post_message"] = test_df["post_message"].astype(str)


# In[6]:


# train_df.head()


# In[7]:


# test_df.head()


# In[8]:


# roberta = '/home/leonard/leonard/nlp/ReINTEL/outputs/pretraining_vlsp'
# roberta_tokenizer = AutoTokenizer.from_pretrained(roberta)
# # roberta_model = TFAutoModel.from_pretrained(roberta)
# roberta_model = TFRobertaModel.from_pretrained(roberta)


# In[9]:


# train_len_word = [len(text.split()) for text in train_df.post_message]
# test_len_word = [len(text.split()) for text in test_df.post_message]
# test_len_char = [len(text) for text in train_df.post_message]
# test_len_char = [len(text) for text in test_df.post_message]


# In[10]:


# def length_plot(lengths):
#     plt.figure(figsize=(12,7))
#     textstr = f' Mean: {np.mean(lengths):.2f} \u00B1 {np.std(lengths):.2f} \n Max: {np.max(lengths)}'
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

#     plt.text(0, 0, textstr, fontsize=14,
#             verticalalignment='top', bbox=props)
#     sns.countplot(lengths)


# In[11]:


# length_plot(train_len_word)


# In[12]:


# length_plot(test_len_word)


# In[13]:


# MAX_LEN = 256
# BATCH_SIZE = 48


# In[14]:


# def regular_encode(texts, maxlen=MAX_LEN):

#     roberta_enc_di = roberta_tokenizer.batch_encode_plus(
#         texts,
#         return_token_type_ids=True,
#         pad_to_max_length=True,
#         max_length=maxlen,
#         truncation=True,
#     )

#     roberta_enc = (
#         np.array(roberta_enc_di["input_ids"]),
#         np.array(roberta_enc_di["attention_mask"]),
#         np.array(roberta_enc_di["token_type_ids"]),
#     )
#     return roberta_enc


# In[15]:


# AUTO = tf.data.experimental.AUTOTUNE

# def data_generator(train_df, val_df):

#     X_train = regular_encode(train_df["post_message"].values, maxlen=MAX_LEN)
#     # y_train = tf.keras.utils.to_categorical(train_df['Label'].values, num_classes=2)
#     y_train = train_df["label"].values
#     X_val = regular_encode(val_df["post_message"].values, maxlen=MAX_LEN)
#     # y_val = tf.keras.utils.to_categorical(val_df['Label'].values, num_classes=2)
#     y_val = val_df["label"].values

#     train_dataset = (
#         tf.data.Dataset.from_tensor_slices((X_train, y_train))
#         .repeat()
#         .shuffle(1024)
#         .batch(BATCH_SIZE)
#         .prefetch(AUTO)
#     )

#     valid_dataset = (
#         tf.data.Dataset.from_tensor_slices((X_val, y_val))
#         .batch(BATCH_SIZE)
#         .cache()
#         .prefetch(AUTO)
#     )

#     return train_dataset, valid_dataset


# In[16]:


# def build_model(bert_model_name_or_path, max_len=384, n_hiddens=-1):
#     bert_model = TFAutoModel.from_pretrained(bert_model_name_or_path)

#     bert_input_word_ids = tf.keras.layers.Input(
#         shape=(max_len,), dtype=tf.int32, name="bert_input_id"
#     )
#     bert_attention_mask = tf.keras.layers.Input(
#         shape=(max_len,), dtype=tf.int32, name="bert_attention_mask"
#     )
#     bert_token_type_ids = tf.keras.layers.Input(
#         shape=(max_len,), dtype=tf.int32, name="bert_token_type_ids"
#     )

#     bert_sequence_output = bert_model(
#         bert_input_word_ids,
#         attention_mask=bert_attention_mask,
#         token_type_ids=bert_token_type_ids,
# #         output_hidden_states=True,
# #         output_attentions=True,

#     )

#     # print(len(bert_sequence_output)) # 4

#     # print(bert_sequence_output[0].shape) # (None, max_len, 768)

#     # print(bert_sequence_output[1].shape) # (None, 768)
#     # print(len(bert_sequence_output[2])) # 13
#     # print(bert_sequence_output[2][0].shape) # (None, max_len, 768)
#     # print(len(bert_sequence_output[3])) # 12
#     # print(bert_sequence_output[3][0].shape) # (None, 12, None, max_len)

#     # TODO: get bert embedding

# #     if n_hiddens == -1:  # get [CLS] token embedding only
# #         # print("Get pooler output of shape (batch_size, hidden_size)")
# #         bert_sequence_output = bert_sequence_output[0][:, 0, :]
# #     else:  # concatenate n_hiddens final layer
# #         # print(f"Concatenate {n_hiddens} hidden_states of shape (batch_size, hidden_size)")
# #         bert_sequence_output = tf.concat(
# #             [bert_sequence_output[2][-i] for i in range(n_hiddens)], axis=-1)

#     # print("bert_sequence_output shape", bert_sequence_output.shape)


# # MLP
# #     out = tf.keras.layers.Flatten()(bert_sequence_output)
# #     out = tf.keras.layers.Dense(1, activation="sigmoid")(out)
# # CNN
#     out = tf.keras.layers.Dropout(0.15)(bert_sequence_output[0])
#     out = tf.keras.layers.Conv1D(768, 2,padding='same')(out)
#     out = tf.keras.layers.LeakyReLU()(out)
#     out = tf.keras.layers.Conv1D(64, 2,padding='same')(out)
# #     out = tf.keras.layers.Dense(1)(out)
#     out = tf.keras.layers.Flatten()(out)
#     out = tf.keras.layers.Dense(1)(out)
#     out = tf.keras.layers.Activation('sigmoid')(out)


#     model = tf.keras.models.Model(
#         inputs=[
#             bert_input_word_ids,
#             bert_attention_mask,
#             bert_token_type_ids,  # bert input
#         ],
#         outputs=out,
#     )
#     model.compile(
#         tf.keras.optimizers.Adam(lr=5e-5),
#         loss="binary_crossentropy",
#         metrics=[tf.keras.metrics.AUC()],
#     )

#     return model


# In[17]:


# %%time
# model = build_model(roberta, max_len=MAX_LEN)
# model.summary()


# In[18]:


# n_splits = 5
# n_epochs = 5

# DISPLAY=1 # USE display=1 FOR INTERACTIVE
# exp = f'phobert_cnn_{MAX_LEN}_len'

# output_dir = f'../outputs/{exp}_models'
# os.makedirs(output_dir, exist_ok=True)


# In[19]:


# def scheduler(epoch):
#     return 3e-5*0.2**epoch


# In[20]:


# strategy = tf.distribute.MirroredStrategy()


# In[21]:


# # for fold, (idxT, idxV) in enumerate(kf.split(train_df)):
# for fold in sorted(train_df["fold"].unique()):
# #     if fold<3:
# #         continue
#     print('*'*100)
#     print(f'FOLD: {fold+1}/{n_splits}')

#     K.clear_session()
#     with strategy.scope():
#         model = build_model(roberta, max_len=MAX_LEN)
# #     model = build_model(roberta, max_len=MAX_LEN)

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


# In[22]:


# X_test = regular_encode(test_df['post_message'].values, maxlen=MAX_LEN)
# y_test = np.zeros((len(test_df),1))
# test_dataset = (
#     tf.data.Dataset
#     .from_tensor_slices((X_test,y_test))
#     .batch(BATCH_SIZE)
# )


# In[23]:


# model = build_model(roberta, max_len=MAX_LEN)
# preds = []
# for i, file_name in enumerate(os.listdir(output_dir)):
#     print('_'*80)

#     K.clear_session()
#     model_path = os.path.join(output_dir, file_name)

#     print(f'Inferencing with model from: {model_path}')
#     model.load_weights(model_path)

#     pred = model.predict(test_dataset,
#                          batch_size=128,
#                          verbose=DISPLAY)
#     # print(pred[])
#     preds.append(pred)


# In[24]:


# preds = np.mean(preds, axis=0)
# test_df["prediction"] = preds
# test_df["prediction"].to_csv(f"{exp}.csv", header=False)


# ### EDA

# In[25]:


# %%time

warmup_train_df = pd.read_excel(
    "/home/leonard/leonard/my_work/ReINTEL/data/raw_data/warmup_training_dataset.xlsx", index_col="id"
)
warmup_test_df = pd.read_excel(
    "/home/leonard/leonard/my_work/ReINTEL/data/raw_data/warmup_test_set.xlsx", index_col="id"
)

public_train_df = pd.read_csv("/home/leonard/leonard/my_work/ReINTEL/data/raw_data/public_train.csv")
raw_test_df = pd.read_csv("/home/leonard/leonard/my_work/ReINTEL/data/raw_data/public_test.csv")

# TODO: make use of warmup_test_df
raw_train_df = pd.concat([warmup_train_df, public_train_df]).drop_duplicates()


# In[26]:


raw_train_df.head()


# In[27]:


# %%time

train_df = pd.read_csv("/home/leonard/leonard/my_work/ReINTEL/data/final_data/train_5_folds.csv")
test_df = pd.read_csv("/home/leonard/leonard/my_work/ReINTEL/data/final_data/private_test.csv")
# test_df = pd.read_csv("/home/leonard/leonard/nlp/ReINTEL/data/final_data/test.csv")

train_df["post_message"] = train_df["post_message"].astype(str)
test_df["post_message"] = test_df["post_message"].astype(str)


# In[28]:


def tokenized_text_normalize(text):
    #     text=  re.sub(r'http(\S)+', ' ',text)
    #     text=  re.sub(r'https(\S)+', ' ',text)
    #     text=  re.sub(r'http ...', ' ',text)
    #     text = re.sub(r'@[\S]+',' ',text)
    #     text = text.strip(string.punctuation+" ")
    #     text = re.sub("\*", " ", text)
    #     text = re.sub("#", " ", text)
    #     text = text.strip(r"# *" )
    text = re.sub(" _ ", " ", text)
    #     text = re.sub("\.+",  "\.", text)
    text = re.sub("…", "...", text)
    #     text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"- - -", " ", text)
    text = re.sub("_ ", " ", text)
    text = re.sub(" +", " ", text)
    return text


# In[29]:


# tokenized_text_normalize(input())


# In[30]:


train_df["post_message"] = train_df["post_message"].apply(tokenized_text_normalize)
test_df["post_message"] = test_df["post_message"].apply(tokenized_text_normalize)


# In[31]:


with open("train.txt", "w") as f:
    f.write("\n".join(train_df["post_message"].astype(str)))
with open("test.txt", "w") as f:
    f.write("\n".join(test_df["post_message"].astype(str)))


# In[32]:


train_df.head()


# In[33]:


def extract_num(text):
    if type(text) == str:
        if text == "unknown":
            return math.nan
        return re.findall("\d+", text)[0]
    return text


# In[34]:


for col in ["timestamp_post", "num_like_post", "num_comment_post", "num_share_post"]:
    train_df[col] = train_df[col].apply(extract_num)
    train_df[col] = train_df[col].astype(float)
    train_df[col] = train_df[col].fillna(-train_df[col].mean())
    test_df[col] = test_df[col].apply(extract_num)
    test_df[col] = test_df[col].astype(float)
    test_df[col] = test_df[col].fillna(-test_df[col].mean())


# In[35]:


# nan_dict = {"timestamp_post": 1e10, "num_like_post":-2,"num_comment_post":-2, "num_share_post":-2}
# unknown_dict = {"timestamp_post": 1e10, "num_like_post":-2,"num_comment_post":-2, "num_share_post":-2}


# def vlsp_impute(text, field):
#     if type(text)!=str:
#         if math.isnan(text):
#             return nan_dict[field]
#     elif text=="unknown":
#         return unknown_dict[field]
#     else:
#         try:
#             return int(extract_num(text))
#         except:
#             print(text)
#             return nan_dict[field]


# In[36]:


# scaler = StandardScaler()
# scaler.fit(pd.concat([train_df, test_df])[["timestamp_post", "num_like_post", "num_comment_post", "num_share_post"]].values)
# train_df[["timestamp_post", "num_like_post", "num_comment_post", "num_share_post"]] = scaler.transform(train_df[["timestamp_post", "num_like_post", "num_comment_post", "num_share_post"]])
# test_df[["timestamp_post", "num_like_post", "num_comment_post", "num_share_post"]] = scaler.transform(test_df[["timestamp_post", "num_like_post", "num_comment_post", "num_share_post"]])


# In[37]:


# train_df["timestamp_post"] = train_df["timestamp_post"].fillna(-1e10)
# test_df["timestamp_post"] = test_df["timestamp_post"].fillna(-1e10)


# In[38]:


MAX_LEN = 256
BATCH_SIZE = 48
roberta = "vinai/phobert-base"
roberta_tokenizer = AutoTokenizer.from_pretrained(
    "/home/leonard/leonard/my_work/ReINTEL/pretrained_phobert-base", use_fast=False
)


# In[39]:


def regular_encode(df, max_len=MAX_LEN):

    roberta_enc_di = roberta_tokenizer.batch_encode_plus(
        df["post_message"].values,
        return_token_type_ids=True,
        pad_to_max_length=True,
        max_length=max_len,
        truncation=True,
    )
    timestamp_post = (df["timestamp_post"] / 1e13).values
    num_like_post = (df["num_like_post"] / 1e8).values
    num_comment_post = (df["num_comment_post"] / 1e8).values
    num_share_post = (df["num_share_post"] / 1e8).values

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


# In[40]:


AUTO = tf.data.experimental.AUTOTUNE


def data_generator(train_df, val_df):

    X_train = regular_encode(train_df, max_len=MAX_LEN)
    # y_train = tf.keras.utils.to_categorical(train_df['Label'].values, num_classes=2)
    y_train = train_df["label"].values
    X_val = regular_encode(val_df, max_len=MAX_LEN)
    # y_val = tf.keras.utils.to_categorical(val_df['Label'].values, num_classes=2)
    y_val = val_df["label"].values

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train)).repeat().shuffle(1024).batch(BATCH_SIZE).prefetch(AUTO)
    )

    valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).cache().prefetch(AUTO)

    return train_dataset, valid_dataset


# In[41]:


def build_model(bert_model_name_or_path="vinai/phobert-base", max_len=384, n_hiddens=-1):
    bert_model = TFAutoModel.from_pretrained(bert_model_name_or_path)

    bert_input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="bert_input_id")
    bert_attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="bert_attention_mask")
    bert_token_type_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="bert_token_type_ids")

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
        bert_sequence_output = tf.concat([bert_sequence_output[2][-i] for i in range(n_hiddens)], axis=-1)
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


# In[42]:


# get_ipython().run_cell_magic("time", "", "model = build_model(max_len=MAX_LEN)\nmodel.summary()")
model = build_model(max_len=MAX_LEN)
model.summary()

# In[43]:


tf.keras.utils.plot_model(model)


# In[43]:


n_splits = 5
n_epochs = 5

DISPLAY = 1  # USE display=1 FOR INTERACTIVE
exp = f"phobert+auxiliary_{MAX_LEN}_len"

output_dir = f"../outputs/{exp}_models"
os.makedirs(output_dir, exist_ok=True)


# In[44]:


def scheduler(epoch):
    return 3e-5 * 0.2 ** epoch


# In[45]:


strategy = tf.distribute.MirroredStrategy()


# In[ ]:


# for fold, (idxT, idxV) in enumerate(kf.split(train_df)):
for fold in sorted(train_df["fold"].unique()):
    print("*" * 100)
    print(f"FOLD: {fold+1}/{n_splits}")
    if fold < 4:
        continue
    K.clear_session()
    with strategy.scope():
        model = build_model(max_len=MAX_LEN)

    reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

    model_dir = os.path.join(output_dir, f"Fold_{fold+1}.h5")

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
    train_dataset, valid_dataset = data_generator(train_df_, val_df_)

    n_steps = train_df_.shape[0] // BATCH_SIZE + 1
    train_history = model.fit(
        train_dataset,
        steps_per_epoch=n_steps,
        callbacks=[
            sv,
            reduce_lr,
            # tb
        ],
        validation_data=valid_dataset,
        epochs=n_epochs,
    )


# In[ ]:


# 0.94458 0.94645 0.95718 0.94500


# In[ ]:


X_test = regular_encode(test_df, max_len=MAX_LEN)
y_test = np.zeros((len(test_df), 1))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)


# In[ ]:


model = build_model(max_len=MAX_LEN)
preds = []
for i, file_name in enumerate(os.listdir(output_dir)):
    print("_" * 80)

    K.clear_session()
    model_path = os.path.join(output_dir, file_name)

    print(f"Inferencing with model from: {model_path}")
    model.load_weights(model_path)

    pred = model.predict(test_dataset, batch_size=128, verbose=DISPLAY)
    # print(pred[])
    preds.append(pred)


# In[ ]:


preds = np.mean(preds, axis=0)
test_df["prediction"] = preds
test_df["prediction"].to_csv(f"{exp}.csv", header=False)


# ## pretraining

# In[ ]:


import pandas as pd
from tqdm import tqdm

from data.tokenizer import VnCoreTokenizer

tqdm.pandas()
vncoretokenizer = VnCoreTokenizer()


# In[ ]:


def text_normalize(text):
    text = text.strip()
    text = re.sub("^TTO *- *", "", text)
    text = re.sub("^VOV\.VN *- *", "", text)
    #     text = vncoretokenizer.tokenize(text)
    return text


# In[ ]:


news_df = pd.read_csv("/home/leonard/leonard/nlp/ReINTEL/data/newdata_21112020.csv")[["title", "content"]].dropna()


# In[ ]:


news_df["content"] = news_df["content"].apply(text_normalize)


# In[ ]:


news_df["title"] = news_df["title"].apply(text_normalize)


# In[ ]:


news_df.head(20)


# In[ ]:


# news_df.to_csv("/home/leonard/leonard/nlp/ReINTEL/data/tokenized_news_data.csv", index=False)


# In[ ]:


# news_df = pd.read_csv("/home/leonard/leonard/nlp/ReINTEL/data/tokenized_news_data.csv")
# news_df.head()


# In[ ]:


all_documents = []

for index, row in tqdm(news_df.iterrows(), total=len(news_df)):
    title = vncoretokenizer.tokenize(str(row["title"]))
    text = str(row["content"])
    sentences = vncoretokenizer.tokenize(text, return_sentences=True)
    document = [title] + sentences
    all_documents.append(document)


# In[ ]:


random.shuffle(all_documents)


# In[ ]:


train_documents = all_documents[: int(len(all_documents) * 0.8)]
val_documents = all_documents[int(len(all_documents) * 0.8) :]
print(len(train_documents), len(val_documents))


# In[ ]:


# with open("/home/leonard/leonard/nlp/ReINTEL/data/tokenized_news_train.txt", "w") as f:
#     for document in tqdm(train_documents):
#         for sentence in document:
#             f.write(sentence)
#             f.write("\n")
#         f.write("\n")


# In[ ]:


# with open("/home/leonard/leonard/nlp/ReINTEL/data/tokenized_news_val.txt", "w") as f:
#     for document in tqdm(val_documents):
#         for sentence in document:
#             f.write(sentence)
#             f.write("\n")
#         f.write("\n")


# In[ ]:


warmup_train_df = pd.read_excel(
    "/home/leonard/leonard/nlp/ReINTEL/data/raw_data/warmup_training_dataset.xlsx", index_col="id"
)
warmup_test_df = pd.read_excel("/home/leonard/leonard/nlp/ReINTEL/data/raw_data/warmup_test_set.xlsx", index_col="id")

public_train_df = pd.read_csv("/home/leonard/leonard/nlp/ReINTEL/data/raw_data/public_train.csv")
test_df = pd.read_csv("/home/leonard/leonard/nlp/ReINTEL/data/raw_data/public_test.csv")

# TODO: make use of warmup_test_df
train_df = pd.concat([warmup_train_df, public_train_df]).drop_duplicates()


# In[ ]:


train_documents = train_df["post_message"].values
test_documents = test_df["post_message"].values


# In[ ]:


train_all_documents = []

for index, row in tqdm(train_df.iterrows(), total=len(train_df)):
    text = str(row["post_message"])
    sentences = vncoretokenizer.tokenize(text, return_sentences=True)
    train_all_documents.append(sentences)


# In[ ]:


import random

random.shuffle(train_all_documents)


# In[ ]:


test_all_documents = []

for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
    text = str(row["post_message"])
    sentences = vncoretokenizer.tokenize(text, return_sentences=True)
    test_all_documents.append(sentences)


# In[ ]:


total = len(train_all_documents) + len(test_all_documents)
test_size = int(total * 0.2)
train_vlsp_text = test_all_documents + train_all_documents[:-test_size]
val_vlsp_text = train_all_documents[-test_size:]
print(len(train_vlsp_text), len(val_vlsp_text))


# In[ ]:


with open("/home/leonard/leonard/nlp/ReINTEL/data/tokenized_train_vlsp_text.txt", "w") as f:
    for document in tqdm(train_vlsp_text):
        for sentence in document:
            f.write(sentence)
            f.write("\n")
        f.write("\n")


# In[ ]:


with open("/home/leonard/leonard/nlp/ReINTEL/data/tokenized_val_vlsp_text.txt", "w") as f:
    for document in tqdm(val_vlsp_text):
        for sentence in document:
            f.write(sentence)
            f.write("\n")
        f.write("\n")


# ## RobertaForMaskedLM

# In[ ]:


import tensorflow as tf
import transformers
from transformers import AutoTokenizer, TFAutoModelForMaskedLM

# In[ ]:


MAX_LEN = 128
BATCH_SIZE = 16


# In[ ]:


pretrain_model = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
model = TFAutoModelForMaskedLM.from_pretrained(pretrain_model)


# In[ ]:


def build_model():
    print(f"Using pretrained {pretrain_model}")
    model = TFAutoModelForMaskedLM.from_pretrained(pretrain_model)
    optimizer = tf.keras.optimizers.Adam(lr=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    tf.keras.metrics.SparseCategoricalAccuracy("accuracy")

    model.compile(optimizer, loss=loss, metrics=["accuracy"])
    return model


# In[ ]:


# model = build_model()
# model.summary()


# In[ ]:


from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/home/leonard/leonard/nlp/ReINTEL/data/tokenized_news_train.txt",
    block_size=128,
)


# In[ ]:


from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)


# In[ ]:


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="/home/leonard/leonard/nlp/ReINTEL/outputs",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)


# In[ ]:


## post process


# In[ ]:


best_prediction = pd.read_csv("/home/leonard/leonard/nlp/ReINTEL/vlsp_reintel_text/results.csv", header=None)
test_df = pd.read_csv("/home/leonard/leonard/nlp/ReINTEL/data/raw_data/public_test.csv")


# In[ ]:


best_prediction.head()


# In[ ]:


test_df["prediction"] = best_prediction[1].values


# In[ ]:


test_df.head()


# In[ ]:


def cutoff(prob):
    if prob > 0.92:
        return 1.0
    return prob


# In[ ]:


# for index, row in test_df.iterrows():
#     if "url" in str(row["post_message"]).lower() and len(str(row["post_message"])) < 20:
#         print("*"*50)
#         print(row["post_message"])
#         print(test_df.loc[index, "prediction"])
# #         test_df.loc[index, "prediction"] = 0.005
# #         print(test_df.loc[index, "prediction"])


# In[ ]:


test_df["prediction"].plot()


# In[ ]:


test_df["prediction"].apply(cutoff).to_csv("post_process.csv", header=False)


# In[ ]:


test_df["prediction"].apply(cutoff).plot()
