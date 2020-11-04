import tensorflow as tf 
from transformers import TFAutoModel


def build_model(bert_model_name_or_path="vinai/phobert-base", max_len=384):
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
    )[0]
    # n_hiddens = 4
    # print(len(bert_sequence_output))

    # print(bert_sequence_output[0].shape)
    # print(bert_sequence_output[1].shape)

    # bert_sequence_output = tf.concat([bert_sequence_output[2][-i] for i in range(n_hiddens)] , axis=-1)

    bert_sequence_output = bert_sequence_output[:, 0, :]
    sequence_output = tf.concat(
        [
            bert_sequence_output,

        ],
        axis=-1,
        name="sequence_output",
    )

    out = tf.keras.layers.Flatten()(sequence_output)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(out)

    model = tf.keras.models.Model(
        inputs=[
            bert_input_word_ids,
            bert_attention_mask,
            bert_token_type_ids,  # bert input
        ],
        outputs=out,
    )
    model.compile(
        tf.keras.optimizers.Adam(lr=5e-5),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC()],
    )

    return model