# Tensorflow for Super-resolution deployment

这是一个完整的部署超分辨率模型到安卓设备的解决方案，包含了模型的训练，量化导出，测试，以及部署。

本仓库包含模型的训练，量化导出，测试部分，具体如何将导出模型部署在安卓设备上见[另一仓库](https://github.com/Monaco12138/Android_Ttlite_SR)


## 环境配置
由于tensorflow各个版本之间API接口以及支持的量化算子区别，在该项目中用到的tensorflow有以下三个版本：

* tensorflow2.3 -> 负责模型训练
* tensorflow2.8 -> 负责模型转换为tflite, 模型量化
* tensorflow2.16.1 -> 负责模型推理，最终部署在android上的是这个版本

相应的环境依赖如下：
```yaml
name: tensorflow2.3
dependencies:
- python=3.6
- cudatoolkit=10.1
- pip:
  - tensorflow==2.3.1
  - tensorflow-addons==0.11.2

name: tensorflow2.8
dependencies:
- python=3.9
- cudatoolkit=11.3
- pip:
  - tensorflow-gpu==2.8.0

name: tensorflow2.16
dependencies:
- python=3.11
- pip:
  - tensorflow==2.16.1
  - tensorflow-addons==0.23.0
  - tensorflow-probability==0.24.0
```

## Getting started
我们以[QuickSR](https://openaccess.thecvf.com/content/CVPR2023W/MobileAI/papers/Berger_QuickSRNet_Plain_Single-Image_Super-Resolution_Architecture_for_Faster_Inference_on_Mobile_CVPRW_2023_paper.pdf) 为例，首先我们需要准备模型，以Tensorflow形式重写一遍。

接着需要准备相关的训练数据集，数据集图片组织格式如下：
```yaml
hr_path:
    hr_0001.png
    hr_0002.png
    ...

lr_pathx2:
    lr_0001.png
    lr_0002.png
    ...
```

### Train
```python
import model.quicsr as quicsr
import dataset.dataloader as dataloader

# dataset
train_loader = dataloader.DataLoader(lr_path, hr_path, scale, batch_size, repeat)

# model
model = quicsr.QuicSR(args.scale, num_filter, num_res_blocks)

# optimizer
learning_rate = 1.e-4
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# loss function
loss_function = keras.losses.MeanAbsoluteError()

# training
for epoch in tqdm(range(num_epochs + 1)):
    for step, (lr, hr) in tqdm(enumerate(train_loader.Dataset())):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)
            sr = model(lr, training=True)
            loss = loss_function(hr, sr)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

### Convert
量化导出这部分有两个主要步骤：

* 将动态的(None, None, 3) 输入转换为固定输入格式 (h, w, 3)，如果要保留动态输入的特性，h, w 可填 None，这样导出的模型输入参数为[1,1,1,3]，需要设置输入维度，见 __Test__ 部分
  ```python
  ## 模型必须要以tf.saved_model.load()形式load进来，如果之前保存的是其它形式，需要再保存加载一遍
  model = quicsr.QuicSR(scale, num_filter, num_blocks)
  model.load_weights(save_model_path)
  tf.saved_model.save(save_tmp_path)

  ## the model must be load like that instead of kears.models.Model(). 
  model = tf.saved_model.load(args.model_tmp_path)

  ## setting fixed input size
  h, w = 1080, 1920 
  scale = 2
  concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  concrete_func.inputs[0].set_shape([1, h, w, 3])
  converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
  ```

* 量化，具体可参见tflite[官网](https://www.tensorflow.org/lite/performance/model_optimization)
  ```python
  ## Dynamic range quantization
  converter.optimizations = [tf.lite.Optimize.DEFAULT]

  ## Full integer quantization
  def representative_dataset():
    for lr, hr in representative_dataloader.Dataset():
      yield [lr]

  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_dataset
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.int8  # or tf.uint8
  converter.inference_output_type = tf.int8  # or tf.uint8

  ## Float16 quantization
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_types = [tf.float16]

  ## write tflite model
  tflite_model = converter.convert()
  with open(args.tflite_path, 'wb') as f:
      f.write(tflite_model)
  ```

### Test
提供两种测试方式，可以比对转换为tflite后模型的推理精度下降程度

* 基于Tensoflow模型测试，见test_tf.py
  ```python
  # dataset
  test_loader = dataloader.DataLoader(lr_path, hr_path, scale, batch_size, repeat)

  # model
  model = quicsr.QuicSR(args.scale, num_filter, num_res_blocks)

  # test
  for step, (lr, hr) in enumerate(test_loader.Dataset()):
    lr = tf.cast(lr, tf.float32)
    hr = tf.cast(hr, tf.float32)
    # model inference
    sr = model.predict(lr)
    sr_image = (sr * 255.0).clip(0, 255).round().astype(np.uint8)
    hr = hr.numpy()
    hr_image = (hr * 255.0).clip(0, 255).round().astype(np.uint8)
    # ...
  ```
* 基于TFLite模型测试，见test_tflite.py
  ```python
    # dataset
    test_loader = dataloader.DataLoader(lr_path, hr_path, scale, batch_size, repeat)

    # test
    for step, (lr, hr) in enumerate(test_loader.Dataset()):
      lr = tf.cast(lr, tf.float32)
      hr = tf.cast(hr, tf.float32)

      # model
      interpreter = tf.lite.Interpreter(model_path=args.model_path)
      input_details = interpreter.get_input_details()
      output_details = interpreter.get_output_details()

      # 设置输入维度
      interpreter.resize_tensor_input(input_details[0]['index'], lr.shape)
      interpreter.allocate_tensors()

      # model inference
      interpreter.set_tensor(input_details[0]['index'], lr)
      interpreter.invoke()

      sr = interpreter.get_tensor(output_details[0]['index'])
      sr_image = (sr * 255.0).clip(0, 255).round().astype(np.uint8)
      hr = hr.numpy()
      hr_image = (hr * 255.0).clip(0, 255).round().astype(np.uint8)
      # ...
    ```
