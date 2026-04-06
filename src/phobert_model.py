from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf

MODEL_NAME = "vinai/phobert-base"

def build_phobert_model(max_len=256):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = TFAutoModel.from_pretrained(MODEL_NAME)

    input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32)

    outputs = encoder(input_ids, attention_mask=attention_mask)[0]
    cls_token = outputs[:, 0, :]

    x = tf.keras.layers.Dense(128, activation="relu")(cls_token)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

    model = tf.keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=outputs
    )

    return tokenizer, model
