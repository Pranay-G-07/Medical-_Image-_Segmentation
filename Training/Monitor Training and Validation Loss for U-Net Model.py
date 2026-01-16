# Import necessary libraries
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Assuming you have already compiled your model
# previous_task_output = model.compile(...)

# Define early stopping and learning rate reduction callbacks
early_stopping = EarlyStopping(
    patience=10,  # Stop training if validation loss doesn't improve for 10 epochs
    restore_best_weights=True  # Restore the best weights found during training
)

learning_rate_scheduler = ReduceLROnPlateau(
    factor=0.5,  # Reduce learning rate by half
    patience=5,   # Reduce after 5 epochs of no improvement
    min_lr=1e-5    # Minimum learning rate
)

# Train the model with callbacks
history = model.fit(
    x_train,  # Training images
    y_train,  # Training masks
    epochs=50,  # Number of epochs to train
    validation_data=(x_val, y_val),  # Validation dataset
    callbacks=[early_stopping, learning_rate_scheduler],  # Monitoring callbacks
    verbose=1  # Print progress bar
)

# Plot training and validation loss
plt.figure(figsize=(10, 5))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Dice Coefficient (if available)
plt.subplot(1, 2, 2)
plt.plot(history.history['dice_coefficient'], label='Training Dice Coefficient')
plt.plot(history.history['val_dice_coefficient'], label='Validation Dice Coefficient')
plt.title('Training and Validation Dice Coefficient')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.legend()

plt.tight_layout()
plt.show()
