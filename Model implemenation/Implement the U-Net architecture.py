from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import backend as K

def encoder_block(inputs, filters):
    # First convolutional layer
    x = Conv2D(filters=filters, kernel_size=(3,3), 
                strides=(1,1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = K.relu(x)
    
    # Second convolutional layer
    x = Conv2D(filters=filters, kernel_size=(3,3), 
                strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = K.relu(x)
    
    # Max pooling layer
    pool = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    
    return x, pool

# Example usage
input_shape = (256, 256, 3)  # Assuming input size of 256x256x3
inputs = Input(shape=input_shape)

# First encoder block
filters = 32
x, pool1 = encoder_block(inputs, filters)

# Second encoder block
filters *= 2
x, pool2 = encoder_block(x, filters)

# Third encoder block
filters *= 2
x, pool3 = encoder_block(x, filters)
