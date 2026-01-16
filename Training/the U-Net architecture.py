def decoder_block(inputs, skip_features, num_filters, kernel_size, stride, padding, dropout_rate):
    """
    Decoder block for U-Net architecture
    Args:
        inputs: Input tensor from the previous layer
        skip_features: Feature maps from the encoder to be concatenated
        num_filters: Number of filters in the convolutional layers
        kernel_size: Size of the convolutional kernels
        stride: Stride for convolutional operations
        padding: Padding type for convolutional layers
        dropout_rate: Dropout rate for regularization
    Returns:
        Concatenated feature maps after upsampling and convolution
    """
    # Upsample the input features
    upsampled = Conv2DTranspose(
        filters=num_filters,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding
    )(inputs)
    
    # Concatenate with skip features from encoder
    concatenated = concatenate([upsampled, skip_features])
    
    # Apply convolution followed by activation
    conv = Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding=padding
    )(concatenated)
    conv = Activation('relu')(conv)
    
    # Apply dropout for regularization
    if dropout_rate > 0:
        conv = Dropout(dropout_rate)(conv)
        
    return conv

def decoder_network(encoder_features, num_filters, kernel_size, stride, padding, dropout_rate):
    """
    Complete decoder network consisting of multiple decoder blocks
    Args:
        encoder_features: List of feature maps from the encoder network
        num_filters: Number of filters for each decoder block
        kernel_size: Size of the convolutional kernels
        stride: Stride for transposed convolution
        padding: Padding type for convolutional layers
        dropout_rate: Dropout rate for regularization
    Returns:
        Output tensor of the decoder network
    """
    decoder_output = encoder_features[-1]  # Start with the last feature map from encoder
    
    # Iterate through the remaining encoder features in reverse order
    for i in range(len(encoder_features)-2, -1, -1):
        decoder_output = decoder_block(
            inputs=decoder_output,
            skip_features=encoder_features[i],
            num_filters=num_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dropout_rate=dropout_rate
        )
    
    return decoder_output
