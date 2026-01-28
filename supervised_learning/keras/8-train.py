#!/usr/bin/env python3
"""
Train a model using mini-batch gradient descent with early stopping,
learning rate decay, and model checkpointing
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """
    Trains a model with early stopping, learning rate decay, and checkpointing
    """
    callbacks = []

    if early_stopping and validation_data is not None:
        early_stop_callback = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=1,
            restore_best_weights=True
        )
        callbacks.append(early_stop_callback)

    if learning_rate_decay and validation_data is not None:
        def schedule(epoch):
            """Inverse time decay learning rate schedule"""
            lr = alpha / (1 + decay_rate * epoch)
            return lr

        lr_scheduler = K.callbacks.LearningRateScheduler(
            schedule=schedule,
            verbose=1
        )
        callbacks.append(lr_scheduler)

    if save_best and filepath is not None and validation_data is not None:
        checkpoint_callback = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=0
        )
        callbacks.append(checkpoint_callback)

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )

    return history
