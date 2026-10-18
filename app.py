from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import tensorflow as tf

from src.config import MODEL_PATH, IMG_SIZE


app = Flask(__name__)


# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# ---------------------------------------------------------
# PREDICTION API
# ---------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Check whether an image was uploaded
    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    # Check whether a file was actually selected
    if file.filename == "":
        return jsonify({
            "error": "No image selected"
        }), 400

    try:

        # Read uploaded image bytes
        image_bytes = np.frombuffer(
            file.read(),
            np.uint8
        )

        # Decode image
        image = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:
            return jsonify({
                "error": "Invalid image file"
            }), 400

        # OpenCV loads images as BGR.
        # Convert to RGB for the TensorFlow model.
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Resize to the model's expected input size
        image = cv2.resize(
            image,
            IMG_SIZE
        )

        # Convert to float32
        image = image.astype(np.float32)

        # Add batch dimension
        image = np.expand_dims(
            image,
            axis=0
        )

        # Run model prediction
        probability = float(
            model.predict(
                image,
                verbose=0
            )[0][0]
        )

        # Convert probability to class prediction
        if probability >= 0.5:

            prediction = "PNEUMONIA"
            confidence = probability

        else:

            prediction = "NORMAL"
            confidence = 1 - probability

        # Return JSON response
        return jsonify({
            "prediction": prediction,
            "model_confidence": round(
                confidence,
                4
            ),
            "disclaimer": (
                "Educational research prototype. "
                "Not a medical diagnosis."
            )
        })

    except Exception as error:

        print("Prediction error:", error)

        return jsonify({
            "error": "An error occurred while processing the image."
        }), 500


# ---------------------------------------------------------
# APPLICATION START
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )