import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.config import MODEL_PATH, IMG_SIZE, PREDICTION_DIR


def predict_image(image_path):
    print("Loading model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading image...")

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(image_rgb, IMG_SIZE)

    input_image = resized.astype(np.float32)
    input_image = np.expand_dims(input_image, axis=0)

    print("Running prediction...")

    probability = float(
        model.predict(input_image, verbose=0)[0][0]
    )

    if probability >= 0.5:
        prediction = "PNEUMONIA"
        confidence = probability
    else:
        prediction = "NORMAL"
        confidence = 1 - probability

    print()
    print("=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"Prediction: {prediction}")
    print(f"Model confidence: {confidence * 100:.2f}%")
    print("=" * 50)

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title(
        f"Prediction: {prediction}\n"
        f"Model confidence: {confidence * 100:.2f}%"
    )

    output_path = PREDICTION_DIR / "prediction_result.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Result saved to:")
    print(output_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("python -m src.predict path/to/image.jpeg")
        sys.exit(1)

    predict_image(sys.argv[1])