# Deep Learning with Keras / TensorFlow Cheat Sheet

> Build, train, and deploy neural networks from scratch to production.

---

## Setup
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers, losses, metrics

print(tf.__version__)
tf.random.set_seed(42)

# GPU check
print("GPU available:", len(tf.config.list_physical_devices("GPU")) > 0)
```

---

## Building Models

### Sequential API (simple stacks)
```python
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax")
], name="my_mlp")

model.summary()
```

### Functional API (multiple inputs/outputs, branches)
```python
inputs = keras.Input(shape=(784,))
x = layers.Dense(256, activation="relu")(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs, outputs, name="functional_mlp")
```

---

## Compile

```python
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",    # or "binary_crossentropy", "mse"
    metrics=["accuracy"]
)
```

| Task | Loss | Output activation |
|------|------|------------------|
| Binary classification | `binary_crossentropy` | `sigmoid` |
| Multiclass (int labels) | `sparse_categorical_crossentropy` | `softmax` |
| Multiclass (one-hot) | `categorical_crossentropy` | `softmax` |
| Regression | `mse` or `huber` | `linear` (none) |
| Multi-label | `binary_crossentropy` | `sigmoid` |

---

## Training

```python
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    shuffle=True,
    verbose=1,
    callbacks=[
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
        callbacks.TensorBoard(log_dir="./logs")
    ]
)
```

---

## Plot Training History

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["loss"], label="Train")
axes[0].plot(history.history["val_loss"], label="Validation")
axes[0].set_title("Loss"); axes[0].legend()

axes[1].plot(history.history["accuracy"], label="Train")
axes[1].plot(history.history["val_accuracy"], label="Validation")
axes[1].set_title("Accuracy"); axes[1].legend()

plt.tight_layout(); plt.show()
```

---

## Evaluation & Inference

```python
loss, acc = model.evaluate(X_test, y_test, verbose=0)
y_pred_prob = model.predict(X_test)
y_pred = y_pred_prob.argmax(axis=1)   # multiclass
y_pred = (y_pred_prob > 0.5).astype(int)  # binary
```

---

## Convolutional Neural Networks (CNNs)

```python
model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),       # H, W, C

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),        # better than Flatten for CNNs
    layers.Dropout(0.5),

    layers.Dense(256, activation="relu"),
    layers.Dense(10, activation="softmax")
])
```

### Data Augmentation
```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1),
])

# Use inside model or as preprocessing
augmented_model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    data_augmentation,
    # ... rest of CNN
])
```

---

## Recurrent Neural Networks (RNNs / LSTMs)

```python
# Sequence classification
model = keras.Sequential([
    layers.Input(shape=(timesteps, features)),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Bidirectional LSTM
layers.Bidirectional(layers.LSTM(64, return_sequences=True))

# GRU (faster than LSTM, often similar accuracy)
layers.GRU(128, return_sequences=True)
```

---

## Transfer Learning

```python
# Load pretrained base (e.g., EfficientNetB0 on ImageNet)
base_model = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
base_model.trainable = False   # freeze base

# Add custom head
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, epochs=10, validation_data=val_ds)

# Fine-tune: unfreeze last N layers
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
model.compile(optimizer=optimizers.Adam(1e-5), ...)  # lower LR for fine-tuning
model.fit(train_ds, epochs=10, validation_data=val_ds)

# Available base models
keras.applications.VGG16(...)
keras.applications.ResNet50(...)
keras.applications.MobileNetV2(...)
keras.applications.EfficientNetB0(...)  # through B7
keras.applications.InceptionV3(...)
```

---

## Custom Training Loop

```python
optimizer = optimizers.Adam(1e-3)
loss_fn = losses.SparseCategoricalCrossentropy()
train_metric = metrics.SparseCategoricalAccuracy()

@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        loss = loss_fn(y_batch, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_metric.update_state(y_batch, predictions)
    return loss

for epoch in range(epochs):
    for x_batch, y_batch in train_dataset:
        loss = train_step(x_batch, y_batch)
    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={train_metric.result():.4f}")
    train_metric.reset_states()
```

---

## Regularization Techniques

```python
# Dropout
layers.Dropout(0.3)            # randomly zeros 30% of neurons

# L1/L2 Weight Regularization
from tensorflow.keras import regularizers
layers.Dense(64, activation="relu",
             kernel_regularizer=regularizers.l2(1e-4),
             bias_regularizer=regularizers.l1(1e-5))

# Batch Normalization (usually before activation)
layers.Dense(64), layers.BatchNormalization(), layers.Activation("relu")

# Layer Normalization (better for RNNs/Transformers)
layers.LayerNormalization()

# Early Stopping (most practical)
callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
```

---

## Save & Load

```python
# Save full model
model.save("my_model.keras")              # recommended format
model = keras.models.load_model("my_model.keras")

# Save weights only
model.save_weights("weights.h5")
model.load_weights("weights.h5")

# SavedModel format (for TF Serving)
model.save("saved_model/")
model = tf.saved_model.load("saved_model/")
```

---

## tf.data Pipeline (for large datasets)

```python
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = (train_ds
    .cache()
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE))

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

model.fit(train_ds, epochs=20, validation_data=val_ds)
```

---

## Common Hyperparameters Guide

| Hyperparameter | Typical start |
|----------------|--------------|
| Learning rate | `1e-3` (Adam), `1e-2` (SGD) |
| Batch size | 32–128 |
| Dropout rate | 0.2–0.5 |
| L2 regularization | 1e-4 |
| Early stopping patience | 5–15 epochs |
| Dense layers | 2–4 hidden layers |
| Neurons per layer | Powers of 2: 64, 128, 256, 512 |
