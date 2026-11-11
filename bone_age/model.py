from bone_age.integrity import verify_weights

BRACKET_NAMES = ["bracket_1_youngest", "bracket_2", "bracket_3", "bracket_4_oldest"]


def build_and_load_model():
    """Rebuild the architecture in code and load only the numeric weights (no code is loaded from files)."""
    import tensorflow as tf

    weights_path = verify_weights()

    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(BRACKET_NAMES), activation="softmax"),
    ])
    model.load_weights(str(weights_path))
    return model
