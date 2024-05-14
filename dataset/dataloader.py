import tensorflow as tf
import numpy as np
import os

class DataLoader:
    def __init__(self, lr_path, hr_path, scale=2, crop_size=48, batch_size=32, augment=True, repeat=64, cache=True):
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.augment = augment
        self.repeat = repeat
        self.cache = cache
        self.scale = scale

    def hr_dataset(self):
        hr_files = self.get_image_file_path(self.hr_path)
        dataset = self.images_dataset(hr_files)
        if self.cache:
            dataset = dataset.cache()
        return dataset

    def lr_dataset(self):
        lr_files = self.get_image_file_path(self.lr_path)
        dataset = self.images_dataset(lr_files)
        if self.cache:
            dataset = dataset.cache()
        return dataset

    def Dataset(self):
        ds = tf.data.Dataset.zip((self.lr_dataset(), self.hr_dataset()))
        
        if self.augment:
            # crop image
            ds = ds.map(lambda lr, hr: crop_image(lr, hr, crop_size=self.crop_size, scale=self.scale), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds = ds.map(random_rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        ds = ds.batch(self.batch_size)
        ds = ds.repeat(self.repeat)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds


    @staticmethod
    def get_image_file_path(root_path):
        filenames = sorted(os.listdir(root_path))
        files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)
            files.append(file)
        return files
    
    @staticmethod
    def images_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # 少了一步，int->float32
        ds = ds.map(lambda x: tf.cast(x, tf.float32) / 255.0, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return ds


def random_rotate(lr, hr):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr, rn), tf.image.rot90(hr, rn)


def random_flip(lr, hr):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr, hr),
                   lambda: (tf.image.flip_left_right(lr),
                            tf.image.flip_left_right(hr)))

def crop_image(lr, hr, crop_size=48, scale=2):
        lr_crop_size = crop_size
        hr_crop_size = crop_size * scale
        lr_image_shape = tf.shape(lr)[:2]

        lr_h = tf.random.uniform(shape=(), maxval=lr_image_shape[0] - lr_crop_size + 1, dtype=tf.int32)
        lr_w = tf.random.uniform(shape=(), maxval=lr_image_shape[1] - lr_crop_size + 1, dtype=tf.int32)

        hr_h = lr_h * scale
        hr_w = lr_w * scale

        lr_cropped = lr[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
        hr_cropped = hr[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]
        
        return lr_cropped, hr_cropped

if __name__ == '__main__':
    hr_path = '/home/ubuntu/data/DIV2K/DIV2K/DIV2K_train_HR1-800'
    lr_path = '/home/ubuntu/data/DIV2K/DIV2K/DIV2K_train_LR_bicubic/X2_1-800'
    train_dataloader = DataLoader(lr_path=lr_path, hr_path=hr_path)
    for lr, hr in train_dataloader.Dataset():
        print(lr.shape)


    
    '''
    ## some examples of tf.data.Dataset ##
    features = np.array([10, 22, 33])
    labels = np.array([1, 0, 1])

    dataset = tf.data.Dataset.from_tensor_slices((features,labels))
    def process(x, y):
        return x * 2, y + 1
    dataset = dataset.map(process)
    for features, labels in dataset:
        print(features, labels)
        print(features.numpy(), labels.numpy())
    '''