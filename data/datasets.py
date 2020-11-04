import tensorflow as tf
import numpy as np

AUTO = tf.data.experimental.AUTOTUNE


def regular_encode(texts, bert_tokenizer, maxlen=256):

    bert_enc_di = bert_tokenizer.batch_encode_plus(
        texts,
        return_token_type_ids=True,
        pad_to_max_length=True,
        max_length=maxlen,
        truncation=True,
    )

    bert_enc = (
        np.array(bert_enc_di["input_ids"]),
        np.array(bert_enc_di["attention_mask"]),
        np.array(bert_enc_di["token_type_ids"]),
    )
    return bert_enc


def data_generator(train_df, val_df, bert_tokenizer, batch_size=32):

    X_train = regular_encode(train_df["post_message"].values, bert_tokenizer)
    # y_train = tf.keras.utils.to_categorical(train_df['Label'].values, num_classes=2)
    y_train = train_df["label"].values
    X_val = regular_encode(val_df["post_message"].values, bert_tokenizer)
    # y_val = tf.keras.utils.to_categorical(val_df['Label'].values, num_classes=2)
    y_val = val_df["label"].values

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .repeat()
        .shuffle(1024)
        .batch(batch_size)
        .prefetch(AUTO)
    )

    valid_dataset = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(batch_size)
        .cache()
        .prefetch(AUTO)
    )

    return train_dataset, valid_dataset

