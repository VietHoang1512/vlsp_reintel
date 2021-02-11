import numpy as np
import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE


def regular_encode(df, bert_tokenizer, max_len):

    bert_enc_di = bert_tokenizer.batch_encode_plus(
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

    bert_enc = (
        np.array(bert_enc_di["input_ids"]),
        np.array(bert_enc_di["attention_mask"]),
        np.array(bert_enc_di["token_type_ids"]),
        timestamp_post,
        num_like_post,
        num_comment_post,
        num_share_post,
        #         np.concatenate([num_like_post, num_comment_post, num_share_post], axis=0)
    )

    return bert_enc


def data_generator(train_df, val_df, bert_tokenizer, max_len, batch_size):

    X_train = regular_encode(train_df, bert_tokenizer, max_len)
    # y_train = tf.keras.utils.to_categorical(train_df['Label'].values, num_classes=2)
    y_train = train_df["label"].values
    X_val = regular_encode(val_df, bert_tokenizer, max_len)
    # y_val = tf.keras.utils.to_categorical(val_df['Label'].values, num_classes=2)
    y_val = val_df["label"].values

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train)).repeat().shuffle(1024).batch(batch_size).prefetch(AUTO)
    )

    valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).cache().prefetch(AUTO)

    return train_dataset, valid_dataset
