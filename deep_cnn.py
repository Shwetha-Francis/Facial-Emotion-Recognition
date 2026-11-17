import tensorflow as tf
import os
import shutil
import random
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# 1. SETTINGS
# ============================================================

dataset_path = "emotions"

train_folder = "train_augmented"
validation_folder = "validation"

num_augmented = 5
batch_size = 16
seed = 42

classes = ["Angry", "happy", "neutral", "sad"]


# ============================================================
# 2. OFFLINE AUGMENTATION
# ============================================================

offline_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.055),

    tf.keras.layers.RandomZoom(0.05),

    tf.keras.layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    ),

    tf.keras.layers.RandomBrightness(0.10),

    tf.keras.layers.RandomContrast(0.10)
])


# ============================================================
# 3. REMOVE OLD TRAINING/VALIDATION FOLDERS
# ============================================================

if os.path.exists(train_folder):
    shutil.rmtree(train_folder)

if os.path.exists(validation_folder):
    shutil.rmtree(validation_folder)

os.makedirs(train_folder)
os.makedirs(validation_folder)


# ============================================================
# 4. SPLIT DATASET
# ============================================================

random.seed(seed)

for class_name in classes:

    input_folder = os.path.join(
        dataset_path,
        class_name
    )

    train_class_folder = os.path.join(
        train_folder,
        class_name
    )

    validation_class_folder = os.path.join(
        validation_folder,
        class_name
    )

    os.makedirs(train_class_folder)
    os.makedirs(validation_class_folder)

    # Get images
    images = [
        image
        for image in os.listdir(input_folder)
        if image.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    # Shuffle
    random.shuffle(images)

    # 80% training / 20% validation
    split_index = int(len(images) * 0.8)

    train_images = images[:split_index]
    validation_images = images[split_index:]

    print("\nClass:", class_name)

    print(
        "Training originals:",
        len(train_images)
    )

    print(
        "Validation originals:",
        len(validation_images)
    )


    # ========================================================
    # 5. CREATE AUGMENTED TRAINING IMAGES
    # ========================================================

    for image_name in train_images:

        image_path = os.path.join(
            input_folder,
            image_name
        )

        # Copy original image
        original_output_path = os.path.join(
            train_class_folder,
            image_name
        )

        shutil.copy2(
            image_path,
            original_output_path
        )

        # Load image
        image = tf.keras.utils.load_img(
            image_path,
            color_mode="rgb",
            target_size=(48, 48)
        )

        image = tf.keras.utils.img_to_array(image)

        # Generate augmented copies
        for i in range(num_augmented):

            image_batch = tf.expand_dims(
                image,
                axis=0
            )

            augmented_image = offline_augmentation(
                image_batch,
                training=True
            )

            augmented_image = tf.squeeze(
                augmented_image,
                axis=0
            )

            augmented_image = tf.clip_by_value(
                augmented_image,
                0,
                255
            )

            base_name = os.path.splitext(
                image_name
            )[0]

            new_name = (
                f"{base_name}_aug_{i + 1}.png"
            )

            output_path = os.path.join(
                train_class_folder,
                new_name
            )

            tf.keras.utils.save_img(
                output_path,
                augmented_image
            )


    # ========================================================
    # 6. VALIDATION IMAGES
    # ========================================================

    for image_name in validation_images:

        image_path = os.path.join(
            input_folder,
            image_name
        )

        output_path = os.path.join(
            validation_class_folder,
            image_name
        )

        shutil.copy2(
            image_path,
            output_path
        )


# ============================================================
# 7. LOAD TRAINING DATA
# ============================================================

train_data = tf.keras.utils.image_dataset_from_directory(

    train_folder,

    image_size=(48, 48),

    color_mode="grayscale",

    batch_size=batch_size,

    shuffle=True,

    seed=seed
)


# ============================================================
# 8. LOAD VALIDATION DATA
# ============================================================

validation_data = tf.keras.utils.image_dataset_from_directory(

    validation_folder,

    image_size=(48, 48),

    color_mode="grayscale",

    batch_size=batch_size,

    shuffle=False
)


print("\nClasses:", train_data.class_names)


# ============================================================
# 9. NORMALIZATION
# ============================================================

normalization = tf.keras.layers.Rescaling(
    1.0 / 255
)

train_data = train_data.map(
    lambda x, y: (normalization(x), y)
)

validation_data = validation_data.map(
    lambda x, y: (normalization(x), y)
)


# ============================================================
# 10. DEEP CNN MODEL
# ============================================================

model = tf.keras.Sequential([

    # ========================================================
    # INPUT
    # ========================================================

    tf.keras.layers.Input(
        shape=(48, 48, 1)
    ),


    # ========================================================
    # BLOCK 1
    # ========================================================

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),


    # ========================================================
    # BLOCK 2
    # ========================================================

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),


    # ========================================================
    # BLOCK 3
    # ========================================================

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),


    # ========================================================
    # BLOCK 4
    # ========================================================

    tf.keras.layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    tf.keras.layers.GlobalAveragePooling2D(),


    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.3
    ),


    # ========================================================
    # OUTPUT
    # ========================================================

    tf.keras.layers.Dense(
        4,
        activation="softmax"
    )
])


# ============================================================
# 11. MODEL SUMMARY
# ============================================================

print("\n========================================")
print("DEEP CNN MODEL")
print("========================================")

model.summary()


# ============================================================
# 12. COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# 13. CALLBACKS
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=8,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=1e-6,

    verbose=1
)


# ============================================================
# 14. TRAIN
# ============================================================

print("\n========================================")
print("TRAINING")
print("========================================")

history = model.fit(

    train_data,

    validation_data=validation_data,

    epochs=30,

    callbacks=[
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# 15. EVALUATE TRAINING DATA
# ============================================================

train_loss, train_accuracy = model.evaluate(

    train_data,

    verbose=0
)


# ============================================================
# 16. EVALUATE VALIDATION DATA
# ============================================================

val_loss, val_accuracy = model.evaluate(

    validation_data,

    verbose=0
)


print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(
    f"Training Loss: {train_loss:.4f}"
)

print(
    f"Training Accuracy: {train_accuracy:.4f}"
)

print(
    f"Validation Loss: {val_loss:.4f}"
)

print(
    f"Validation Accuracy: {val_accuracy:.4f}"
)


# ============================================================
# 17. SAVE MODEL
# ============================================================

model.save(
    "emotion_deep_cnn.keras"
)

print(
    "\nModel saved as emotion_deep_cnn.keras"
)


# ============================================================
# 18. PREDICTIONS
# ============================================================

y_true = []
y_pred = []

for images, labels in validation_data:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)


# ============================================================
# 20. CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Angry",
            "happy",
            "neutral",
            "sad"
        ],
        zero_division=0
    )
)