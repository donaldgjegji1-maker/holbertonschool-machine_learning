#!/usr/bin/env python3
"""Transfer Learning with CIFAR-10 using EfficientNetB0"""

import numpy as np
from tensorflow import keras as K


def preprocess_data(X, Y):
    """
    Pre-processes the data for the model.

    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3) - CIFAR 10 data
        Y: numpy.ndarray of shape (m,) - CIFAR 10 labels

    Returns:
        X_p: preprocessed X
        Y_p: preprocessed Y (one-hot encoded)
    """
    # Preprocess input for EfficientNetB0
    X_p = K.applications.efficientnet.preprocess_input(X.astype('float32'))

    # One-hot encode labels (10 classes)
    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p


if __name__ == '__main__':
    # Load CIFAR-10 data
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    # Preprocess data
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    # -------------------------
    # Build the full model
    # -------------------------

    # Input layer for 32x32 images
    inputs = K.Input(shape=(32, 32, 3))

    # Lambda layer to upscale from 32x32 to 224x224
    x = K.layers.Lambda(
        lambda img: K.backend.resize_images(
            img,
            height_factor=7,
            width_factor=7,
            data_format='channels_last',
            interpolation='bilinear'
        )
    )(inputs)

    # Load EfficientNetB0 without top, pretrained on ImageNet
    base_model = K.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=x,
        pooling='avg'
    )

    # Freeze ALL base model layers initially
    base_model.trainable = False

    # Add custom classification head
    base_output = base_model.output
    x2 = K.layers.BatchNormalization()(base_output)
    x2 = K.layers.Dense(256, activation='relu')(x2)
    x2 = K.layers.Dropout(0.4)(x2)
    x2 = K.layers.Dense(10, activation='softmax')(x2)

    model = K.Model(inputs=inputs, outputs=x2)

    # -------------------------
    # Phase 1: Pre-compute frozen features (Hint 3 optimization)
    # -------------------------
    feature_extractor = K.Model(inputs=model.input, outputs=base_model.output)

    print("Computing features for training set...")
    train_features = feature_extractor.
    predict(X_train_p, batch_size=128, verbose=1)
    print("Computing features for test set...")
    test_features = feature_extractor
    .predict(X_test_p, batch_size=128, verbose=1)

    # Build a lightweight top model on pre-computed features
    feat_input = K.Input(shape=(train_features.shape[1],))
    y = K.layers.BatchNormalization()(feat_input)
    y = K.layers.Dense(256, activation='relu')(y)
    y = K.layers.Dropout(0.4)(y)
    y = K.layers.Dense(10, activation='softmax')(y)

    top_model = K.Model(inputs=feat_input, outputs=y)
    top_model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n=== Phase 1: Training classification head on frozen features ===")
    top_model.fit(
        train_features, Y_train_p,
        batch_size=128,
        epochs=25,
        validation_data=(test_features, Y_test_p),
        callbacks=[
            K.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy', factor=0.5,
                patience=3, verbose=1),
            K.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=8,
                restore_best_weights=True, verbose=1)
        ]
    )

    # Transfer top_model weights into the full model's top layers
    model_top_layers = [l for l in model.layers if l.name in
                        [tl.name for tl in top_model.layers]]
    for tl in top_model.layers:
        if tl.get_weights():
            try:
                model.get_layer(tl.name).set_weights(tl.get_weights())
            except Exception:
                pass

    # -------------------------
    # Phase 2: Fine-tune last layers of base model
    # -------------------------
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n=== Phase 2: Fine-tuning EfficientNetB0 top layers ===")

    datagen = K.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10,
        zoom_range=0.1
    )
    datagen.fit(X_train_p)

    model.fit(
        datagen.flow(X_train_p, Y_train_p, batch_size=64),
        steps_per_epoch=len(X_train_p) // 64,
        epochs=30,
        validation_data=(X_test_p, Y_test_p),
        callbacks=[
            K.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy', factor=0.5,
                patience=3, min_lr=1e-7, verbose=1),
            K.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=8,
                restore_best_weights=True, verbose=1),
            K.callbacks.ModelCheckpoint(
                'cifar10.h5', monitor='val_accuracy',
                save_best_only=True, verbose=1)
        ]
    )

    model.save('cifar10.h5')
    print("\nModel saved as cifar10.h5")

    loss, acc = model.evaluate(X_test_p, Y_test_p, batch_size=128, verbose=1)
    print(f"\nFinal Test Accuracy: {acc:.4f}")
