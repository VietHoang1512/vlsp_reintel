import os
import pandas as pd

from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras import backend as K
from transformers import AutoTokenizer, TFAutoModel

from data.datasets import regular_encode, data_generator
from data.tokenizer import VnCoreTokenizer
from trainer.model import build_model
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

SEED = 1710
MAX_LEN = 256
N_HIDDENS = -1
BATCH_SIZE = 48
N_SPLITS = 5
N_EPOCHS = 5
DISPLAY = 1  # USE display=1 FOR INTERACTIVE

exp = f'phobert_{MAX_LEN}_len_{N_SPLITS}_folds_{N_HIDDENS}_hidden_states'

train_df = pd.read_csv("../data/tokenized_data/train_5_folds.csv")
test_df = pd.read_csv("../data/tokenized_data/test.csv")

train_df["post_message"] = train_df["post_message"].astype(str)
test_df["post_message"] = test_df["post_message"].astype(str)
bert = "/home/leonard/leonard/vlsp/ReINTEL/pretrained_phobert-base"
# bert = 'vinai/phobert-base'

bert_tokenizer = AutoTokenizer.from_pretrained(bert)
# for SEED in range(10):
#     seed_all(SEED)

#     model = build_model(bert, max_len=MAX_LEN, n_hiddens=N_HIDDENS)


#     model.summary()
#     

#     # kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
#     output_dir = f'../outputs/{exp}_models'
#     os.makedirs(output_dir, exist_ok=True)

#     def scheduler(epoch):
#         return 3e-5*0.2**epoch

#     strategy = tf.distribute.MirroredStrategy()

#     # for fold, (idxT, idxV) in enumerate(kf.split(train_df)):
#     for fold in sorted(train_df["fold"].unique()):
#         print('*'*100)
#         print(f'FOLD: {fold+1}/{N_SPLITS}')
        
#         K.clear_session()
#         with strategy.scope():
#             model = build_model(bert, max_len=MAX_LEN, n_hiddens=N_HIDDENS)

#         reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

#         model_dir = os.path.join(output_dir, f'Fold_{fold+1}_{SEED}.h5')

#         sv = tf.keras.callbacks.ModelCheckpoint(model_dir,
#                                                 monitor='val_auc',
#                                                 verbose=1,
#                                                 save_best_only=True,
#                                                 save_weights_only=True,
#                                                 mode='max',
#                                                 save_freq='epoch')
        
#     #     train_df_ = train_df.iloc[idxT]
#     #     val_df_ = train_df.iloc[idxV]
#         train_df_ = train_df[train_df["fold"]!=fold]
#         val_df_ = train_df[train_df["fold"]==fold]

#         train_dataset, valid_dataset = data_generator(train_df_, val_df_, bert_tokenizer, max_len=MAX_LEN)

#         n_steps = train_df_.shape[0] // BATCH_SIZE
#         train_history = model.fit(
#             train_dataset,
#             steps_per_epoch=n_steps,

#             callbacks=[sv,
#                     reduce_lr,
#                     # tb
#                     ],
#             validation_data=valid_dataset,
#             epochs=N_EPOCHS
#         )
output_dir = "/home/leonard/leonard/vlsp/ReINTEL/outputs/phobert_256_len_5_folds_-1_hidden_states_models"
X_test = regular_encode(test_df['post_message'].values, bert_tokenizer, max_len=MAX_LEN)
y_test = np.zeros((len(test_df), 1))
test_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_test, y_test))
    .batch(BATCH_SIZE)
)
model = build_model(bert, max_len=MAX_LEN, n_hiddens=N_HIDDENS)
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