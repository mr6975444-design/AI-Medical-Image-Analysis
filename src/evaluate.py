import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from src.data_loader import load_datasets

from src.config import (
    MODEL_PATH,
    PLOT_DIR,
    METRICS_DIR
)


def evaluate_model():

    print("=" * 60)
    print("LOADING TEST DATASET")
    print("=" * 60)

    _, _, test_dataset = load_datasets()

    print("\nLoading trained model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded successfully.")

    print("\n")
    print("=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)

    results = model.evaluate(
        test_dataset,
        verbose=1
    )

    print("\nTest Results:")
    print(results)

    # --------------------------------------------------------
    # GET TRUE LABELS
    # --------------------------------------------------------

    y_true = np.concatenate(
        [
            labels.numpy()
            for _, labels in test_dataset
        ]
    ).astype(int).flatten()

    # --------------------------------------------------------
    # GET MODEL PREDICTIONS
    # --------------------------------------------------------

    probabilities = model.predict(
        test_dataset,
        verbose=1
    ).flatten()

    y_pred = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "NORMAL",
            "PNEUMONIA"
        ]
    )

    print("\n")
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(report)

    # Save report
    report_path = (
        METRICS_DIR /
        "classification_report.txt"
    )

    with open(
        report_path,
        "w"
    ) as file:

        file.write(report)

    print(
        "Classification report saved to:"
    )

    print(report_path)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\n")
    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "NORMAL",
            "PNEUMONIA"
        ]
    )

    display.plot()

    plt.title(
        "Medical Image Classification - Confusion Matrix"
    )

    confusion_matrix_path = (
        PLOT_DIR /
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_matrix_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nConfusion matrix saved to:"
    )

    print(confusion_matrix_path)

    print("\n")
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_model()