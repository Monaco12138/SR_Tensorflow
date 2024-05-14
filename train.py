import tensorflow as tf
import dataset.dataloader as dataloader
import argparse
import tensorflow.keras as keras
from tqdm import tqdm
import os
import model.edsr as edsr
import model.quicsr as quicsr

# args
parser = argparse.ArgumentParser()
parser.add_argument('--lr_path', default='/home/ubuntu/data/DIV2K/DIV2K/DIV2K_train_LR_bicubic/X2_1-800')
parser.add_argument('--hr_path', default='/home/ubuntu/data/DIV2K/DIV2K/DIV2K_train_HR1-800')
parser.add_argument('--save_path', default='./save/test')
parser.add_argument('--scale', default=2)
args = parser.parse_args()

# dataset
batch_size = 16
repeat = 64
train_loader = dataloader.DataLoader(lr_path=args.lr_path, hr_path=args.hr_path, scale=args.scale, batch_size=batch_size, repeat=repeat)

# model
num_filter = 32
num_res_blocks = 2
#model = edsr.EDSR(args.scale, num_filter, num_res_blocks)
model = quicsr.QuicSR(args.scale, num_filter, num_res_blocks)
# optimizer
learning_rate = 1.e-4
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss_function = keras.losses.MeanAbsoluteError()

# train
num_epochs = 200
epoch_save = 50
log_interval = 100
loss_mean = keras.metrics.Mean()

for epoch in tqdm(range(num_epochs + 1)):
    # each epoch

    for step, (lr, hr) in tqdm(enumerate(train_loader.Dataset())):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = model(lr, training=True)
            loss = loss_function(hr, sr)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        loss_mean(loss)

        if step % log_interval == 0:
            print(f'[Epoch:{epoch}/step:{step}], loss: {loss_mean.result().numpy():.3f}')
            loss_mean.reset_states()

    if epoch % epoch_save == 0:
        model.save_weights(os.path.join(args.save_path, f'epoch_{epoch}.weights.h5'))