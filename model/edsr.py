import tensorflow as tf
from tensorflow.keras import layers, models

def ResBlock(x_in, num_filter):
    x = layers.Conv2D(num_filter, 3, padding='same', activation='relu')(x_in)
    x = layers.Conv2D(num_filter, 3, padding='same')(x)
    x = layers.Add()([x_in, x])
    return x

def Pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def Upsampler(x, scale, num_filter):
    def upsample_1(x, factor, **kwargs):
        x = layers.Conv2D(num_filter * (factor**2), 3, padding='same', **kwargs)(x)
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

def EDSR(scale, num_filter=64, num_res_blocks=8):
    x_in = layers.Input(shape=(None, None, 3))
    
    x = b = layers.Conv2D(num_filter, 3, padding='same')(x_in)

    for _ in range(num_res_blocks):
        b = ResBlock(b, num_filter)
    b = layers.Conv2D(num_filter, 3, padding='same')(b)
    x = layers.Add()([x,b])

    x = Upsampler(x, scale, num_filter)
    x = layers.Conv2D(3, 3, padding='same')(x)
    return models.Model(x_in, x, name='EDSR')


if __name__ == '__main__':
    model = EDSR(2, 64, 8)
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')

    import numpy as np
    h, w = 540, 960
    scale = 2
    input_ = np.ones((1, h, w, 3)).astype('float32')
    output_ = np.ones((1, h*2, w*2, 3)).astype('float32')
    model.fit(input_, output_, epochs=1)

    sr = model.predict(input_)
    print(sr.shape)