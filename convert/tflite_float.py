import numpy as np
import tensorflow as tf
from PIL import Image

# interpreter = tf.lite.Interpreter(model_path='./quicsr_smallx2.tflite')
interpreter = tf.lite.Interpreter(model_path='./save/quicsr_tflite/quicsr_float32_1080p.tflite')

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

input_scale, input_zero_point = input_details[0]['quantization']
print(input_scale, input_zero_point)


h, w = input_details[0]['shape'][-1], input_details[0]['shape'][-2]

# 输入时，先归一化到[0,1] 再量化为 int8

input_image = np.array(Image.open('./photo/1080p.png').convert('RGB'))#.transpose((2,0,1))#.astype(np.float32)
input_tensor = (input_image / 255.0).astype(np.float32).clip(0, 1)
#
input_tensor = np.expand_dims(input_tensor, axis=0)
print(input_tensor.shape) # (b, c, h, w)

interpreter.set_tensor(input_details[0]['index'], input_tensor)
interpreter.invoke()

output_tensor = interpreter.get_tensor(output_details[0]['index'])

# 输出结果 先转float32, 再反量化为[0,1] 再转 [0,255]

output_image = (output_tensor * 255.0).clip(0, 255).round().astype(np.uint8)#.transpose((0,2,3,1))
output_image = output_image[0]
print(output_image.shape) #(h, w, c), RGB
output_image = Image.fromarray(output_image)
output_image.save('./photo/sr_4k.png')