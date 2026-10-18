import tensorflow as tf

from src.config import IMG_SIZE


def build_model():
    """
    Build a binary image classification model
    using MobileNetV2 transfer learning.
    """

    # --------------------------------------------------------
    # 1. Load pretrained MobileNetV2
    # --------------------------------------------------------

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers initially
    base_model.trainable = False

    # --------------------------------------------------------
    # 2. Define input
    # --------------------------------------------------------

    inputs = tf.keras.Input(
        shape=IMG_SIZE + (3,)
    )

    # --------------------------------------------------------
    # 3. MobileNetV2 preprocessing
    # --------------------------------------------------------

    x = tf.keras.applications.mobilenet_v2.preprocess_input(
        inputs
    )

    # --------------------------------------------------------
    # 4. Feature extraction
    # --------------------------------------------------------

    x = base_model(
        x,
        training=False
    )

    # --------------------------------------------------------
    # 5. Convert feature maps into a vector
    # --------------------------------------------------------

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # --------------------------------------------------------
    # 6. Dropout to reduce overfitting
    # --------------------------------------------------------

    x = tf.keras.layers.Dropout(0.3)(x)

    # --------------------------------------------------------
    # 7. Binary classification output
    # --------------------------------------------------------

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    # --------------------------------------------------------
    # 8. Create model
    # --------------------------------------------------------

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    # --------------------------------------------------------
    # 9. Compile model
    # --------------------------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            )
        ]
    )

    return model