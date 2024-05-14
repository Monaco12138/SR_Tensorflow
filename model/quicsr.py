import tensorflow as tf
from tensorflow.keras import layers, models, activations


def Pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def Upsampler(x, scale, out_channels=3):
    def upsample_1(x, factor, **kwargs):
        x = layers.Conv2D(out_channels * (factor**2), 3, padding='same', **kwargs)(x)
        x = activations.relu(x, max_value=1.0)
        return layers.Lambda(Pixel_shuffle(scale=factor))(x)
    
    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')
    else:
        raise ValueError('Scale must between 2 to 4!')
    return x


def QuicSR(scale, num_filter, num_blocks):
    x_in = layers.Input(shape=(None, None, 3))

    x = layers.Conv2D(num_filter, 3, padding='same')(x_in)
    x = activations.relu(x, max_value=1.0)

    for _ in range(num_blocks):
        x = layers.Conv2D(num_filter, 3, padding='same')(x)
        x = activations.relu(x, max_value=1.0)

    x = Upsampler(x, scale)

    model = models.Model(x_in, x, name='QuicSR') 

    # Initial
    for conv_layer in model.layers:
        if isinstance(conv_layer, layers.Conv2D):
            # Initialise each conv layer so that it behaves similarly to: 
            # y = conv(x) + x after initialization
            if 'scale' not in conv_layer.name:
                middle = conv_layer.kernel_size[0] // 2
                num_residual_channels = min(conv_layer.input_shape[-1], conv_layer.filters)
                weights, biases = conv_layer.get_weights()
                for idx in range(num_residual_channels):
                    weights[middle, middle, idx, idx] += 1.0
                conv_layer.set_weights([weights, biases])
            
            # This will initialize the weights of the last conv so that it behaves like:
            # y = conv(x) + repeat_interleave(x, scaling_factor ** 2) after initialization
            else:
                middle = conv_layer.kernel_size[0] // 2
                out_channels = conv_layer.filters
                scaling_factor_squarred = out_channels // 3
                weights, biases = conv_layer.get_weights()
                for idx_out in range(out_channels):
                    idx_in = (idx_out % out_channels) // scaling_factor_squarred
                    weights[middle, middle, idx_in, idx_out] += 1.0
                conv_layer.set_weights([weights, biases])
    return model


if __name__ == '__main__':
    model = QuicSR(2, 32, 2)
    model.summary()



