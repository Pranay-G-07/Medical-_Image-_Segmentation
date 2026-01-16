from tensorflow.keras import layers, Model, optimizers

# Define the loss function
loss_fn = 'categorical_crossentropy'

# Define the optimizer
optimizer = optimizers.Adam(learning_rate=1e-4)

# Compile the model
model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=['accuracy', 'precision', 'recall']
)
