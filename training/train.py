import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight

DATASET_PATH = "dataset/"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_DIR = "saved_model"
MODEL_PATH = os.path.join(MODEL_DIR, "solar_panel_model.h5")


def get_generators():
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2,
    )

    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )

    val_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
    )

    return train_generator, val_generator


def compute_class_weights(train_generator):
    class_labels = train_generator.classes

    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(class_labels),
        y=class_labels,
    )

    return dict(zip(np.unique(class_labels), weights))


def build_model(num_classes):
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    return model, base_model


def main():
    train_gen, val_gen = get_generators()
    class_weights = compute_class_weights(train_gen)

    model, base_model = build_model(train_gen.num_classes)

    # Stage 1: Train top layers
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_gen,
        epochs=5,
        validation_data=val_gen,
        class_weight=class_weights,
    )

    # Stage 2: Fine-tuning
    base_model.trainable = True

    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        class_weight=class_weights,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)

    print("Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    main()
