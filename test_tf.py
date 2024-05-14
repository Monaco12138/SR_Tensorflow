import tensorflow as tf
import dataset.dataloader as dataloader
import model.edsr as edsr
import model.quicsr as quicsr
import tensorflow.keras as keras
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm

# args
parser = argparse.ArgumentParser()
parser.add_argument('--lr_path', default='/home/ubuntu/data/DIV2K/DIV2K/DIV2K_train_LR_bicubic/X2_801-900')
parser.add_argument('--hr_path', default='/home/ubuntu/data/DIV2K/DIV2K/DIV2K_valid_HR801-900')
parser.add_argument('--model_weight_path', default='./save/quicsr/epoch_200.weights.h5')
parser.add_argument('--save_path', default='./save/test')
parser.add_argument('--scale', default=2)
args = parser.parse_args()

# dataset
batch_size = 1
repeat = 1
test_loader = dataloader.DataLoader(lr_path=args.lr_path, 
                                    hr_path=args.hr_path,
                                    scale=args.scale,
                                    batch_size=batch_size,
                                    repeat=repeat,
                                    augment=False
                                    )

# model
num_filter = 32
num_blocks = 2
model = quicsr.QuicSR(2, num_filter, num_blocks)
model.load_weights(args.model_weight_path)

def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)


psnr_list = []
for step, (lr, hr) in enumerate(test_loader.Dataset()):
    lr = tf.cast(lr, tf.float32)
    hr = tf.cast(hr, tf.float32)

    # model
    sr = model.predict(lr)

    sr_image = (sr * 255.0).clip(0, 255).round().astype(np.uint8)

    hr = hr.numpy()
    hr_image = (hr * 255.0).clip(0, 255).round().astype(np.uint8)

    psnr_val = psnr(sr_image, hr_image)[0].numpy()
    print(f'[{step}/100]: {psnr_val}')
    psnr_list.append(psnr_val)

print("Average psnr on dataset:")
psnr_list = np.array(psnr_list)
print(np.mean(psnr_list))
np.savetxt('./psnr/psnr_tf.txt', psnr_list)
