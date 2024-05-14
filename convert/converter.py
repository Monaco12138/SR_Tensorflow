import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import argparse
import sys
sys.path.append('..')
import model.edsr as edsr
import model.quicsr as quicsr
import dataset.dataloader as dataloader


# args
parser = argparse.ArgumentParser()
parser.add_argument('--lr_path', default='/home/ubuntu/data/DIV2K/DIV2K/DIV2K_train_LR_bicubic/X2_801-900')
parser.add_argument('--hr_path', default='/home/ubuntu/data/DIV2K/DIV2K/DIV2K_valid_HR801-900')
parser.add_argument('--model_weight_path', default='./save/quicsr/epoch_200.weights.h5')
parser.add_argument('--model_tmp_path', default='./save/quicsr_tflite/model_epoch_200')
parser.add_argument('--tflite_path', default='./save/quicsr_tflite/quicsr_float32_test.tflite')
parser.add_argument('--scale', default=2)
args = parser.parse_args()

## model
num_filter = 32
num_res_blocks = 2
# model = edsr.EDSR(2, num_filter, num_res_blocks)
model = quicsr.QuicSR(2, num_filter, num_res_blocks)
model.load_weights(args.model_weight_path)
tf.saved_model.save(model, args.model_tmp_path)

## the model must be load like that instead of kears.models.Model(). 
model = tf.saved_model.load(args.model_tmp_path)

## setting fixed input size
h, w = 1080, 1920
scale = 2
concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([1, h, w, 3])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

# dataset
representative_dataloader = dataloader.DataLoader(lr_path=args.lr_path, hr_path=args.hr_path, crop_size=100, batch_size=1, repeat=1)

def representative_dataset():
    for _ in range(3):
        data = np.random.rand(1, h, w, 3)
        yield [data.astype(np.float32)]
# def representative_dataset():
#     for lr, hr in representative_dataloader.Dataset():
#         yield [lr]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.target_spec.supported_types = [tf.float16]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.target_spec.supported_types = []
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8 
# converter.experimental_new_quantizer = False

tflite_model = converter.convert()


with open(args.tflite_path, 'wb') as f:
    f.write(tflite_model)


interpreter = tf.lite.Interpreter(model_path=args.tflite_path)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)