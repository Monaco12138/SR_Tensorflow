# Tensorflow for Super-resolution deployment

这是一个完整的部署超分辨率模型到安卓设备的解决方案，包含了模型的训练，测试，量化导出以及部署。

本仓库包含模型的训练，测试，量化导出部分，具体如何将导出模型部署在安卓设备上见[另一仓库](https://github.com/Monaco12138/Android_Ttlite_SR)


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

## Introduction
