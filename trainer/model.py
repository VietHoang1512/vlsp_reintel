import tensorflow as tf
from transformers import TFAutoModel, AutoConfig


def scheduler(epoch):
    return 3e-5 * 0.2 ** epoch


def build_model(bert_model_name_or_path, max_len=384, n_hiddens=-1):
    config = AutoConfig.from_pretrained(bert_model_name_or_path, output_attentions=True,output_hidden_states=True,use_cache=True)
    bert_model = TFAutoModel.from_config(config)

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

    # TODO: get bert embedding

    if n_hiddens == -1:  # get [CLS] token embedding only
        print("Get pooler output of shape (batch_size, hidden_size)")
        bert_sequence_output = bert_sequence_output[0][:, 0, :]
    #         bert_sequence_output = bert_sequence_output[1]
    else:  # concatenate n_hiddens final layer
        print(f"Concatenate {n_hiddens} hidden_states of shape (batch_size, hidden_size)")
        bert_sequence_output = tf.concat([bert_sequence_output[2][-i] for i in range(n_hiddens)], axis=-1)
        bert_sequence_output = bert_sequence_output[:, 0, :]

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
