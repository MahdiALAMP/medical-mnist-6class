import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = os.path.join("data", "medical_mnist")  # path to folder with 6 class subfolders
IMG_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 15
SEED = 42
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# DATA LOADING
# -----------------------
def load_datasets():
    """Load train and test datasets using an 80/20 split.

    The directory structure under DATA_DIR is expected to be:

    data/medical_mnist/
        AbdomenCT/
        BreastMRI/
        ChestCT/
        CXR/
        Hand/
        HeadCT/
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="training",
        seed=SEED,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
    )

    class_names = train_ds.class_names
    print("Class names:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (
        train_ds
        .shuffle(1000)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )
    test_ds = (
        test_ds
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )

    return train_ds, test_ds, class_names

# -----------------------
# MODEL
# -----------------------
def build_model(input_shape, num_classes):
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    inputs = layers.Input(shape=input_shape)

    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="medical_mnist_cnn")
    return model

# -----------------------
# PLOTTING
# -----------------------
def plot_history(history, out_path):
    # Accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.savefig(os.path.join(out_path, "accuracy_curve.png"), bbox_inches="tight")
    plt.close()

    # Loss
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(out_path, "loss_curve.png"), bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, class_names, out_path, normalize=False, filename=None):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if not filename:
        filename = "confusion_matrix_normalized.png" if normalize else "confusion_matrix.png"
    plt.savefig(os.path.join(out_path, filename), bbox_inches="tight")
    plt.close()

# -----------------------
# TRAIN + EVAL PIPELINE
# -----------------------
def main():
    train_ds, test_ds, class_names = load_datasets()
    num_classes = len(class_names)

    input_shape = IMG_SIZE + (1,)
    model = build_model(input_shape, num_classes)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(OUTPUT_DIR, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    plot_history(history, OUTPUT_DIR)

    # Evaluate on the held-out test set
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}, test_loss: {test_loss:.4f}")

    # Predictions for metrics
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    print(report)

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nTest accuracy: {test_acc:.4f}, test_loss: {test_loss:.4f}\n")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, OUTPUT_DIR, normalize=False, filename="confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, OUTPUT_DIR, normalize=True, filename="confusion_matrix_normalized.png")


if __name__ == "__main__":
    main()
