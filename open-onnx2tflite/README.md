# open-onnx2tflite

将公开 ONNX 模型转换为 TFLite 格式，并执行 INT8 量化，验证转换精度。

---

## 目录

- [项目结构](#项目结构)
- [脚本说明](#脚本说明)
- [快速开始](#快速开始)
- [关键代码定位](#关键代码定位)
- [INT8 量化原理](#int8-量化原理)
- [验证结果](#验证结果)
- [参考资料](#参考资料)

---

## 项目结构

```
open-onnx2tflite/
├── MODELS_AND_DATASETS.md      # 模型与数据集下载指南
├── README.md                   # 本文件
├── requirements.txt            # Python 依赖
├── download_models.py          # 下载 ONNX 模型
├── download_data.py            # 下载数据集（MNIST + Imagenette）
├── build_calib.py              # 生成 INT8 校准数据
├── convert.py                  # ONNX → TFLite 转换（float32 + INT8）
├── validate.py                 # 转换精度验证
├── models/                     # 模型文件
│   ├── *.onnx                  # 原始 ONNX 模型
│   ├── *_sim.onnx              # 简化后的 ONNX 模型
│   ├── *_float32/              # float32 TFLite 输出
│   └── *_int8/                 # INT8 TFLite 输出
└── data/                       # 数据集与校准文件
    ├── mnist/                  # MNIST 数据集
    ├── imagenette2-320/        # Imagenette 320px 数据集
    ├── calib_mnist_28x28x1_float32.npy       # MNIST 校准数据
    └── calib_imagenet_224x224x3_float32.npy  # ImageNet 校准数据
```

---

## 脚本说明

### `download_models.py` — 下载 ONNX 模型

从 HuggingFace `onnxmodelzoo` 下载 4 个公开 ONNX 模型到 `models/`：

| 模型 | 输入形状 | 大小 |
|------|----------|------|
| MNIST-12 | `[1, 1, 28, 28]` | ~26 KB |
| SqueezeNet1.1-7 | `[1, 3, 224, 224]` | ~5 MB |
| MobileNetV2-7 | `[1, 3, 224, 224]` | ~14 MB |
| ResNet18-v1-7 | `[1, 3, 224, 224]` | ~45 MB |

```bash
python download_models.py                # 下载全部
python download_models.py --model mnist-12  # 下载指定模型
python download_models.py --mirror       # 使用 hf-mirror 国内镜像
```

### `download_data.py` — 下载数据集

下载用于精度验证和 INT8 量化校准的真实数据集：

- **MNIST**：从 AWS 公开镜像下载 idx 文件到 `data/mnist/`
- **Imagenette 320px**：从 fastai S3 下载 ImageNet 10 类子集到 `data/imagenette2-320/`（免注册）

```bash
python download_data.py                  # 下载全部
python download_data.py --dataset mnist  # 仅下载 MNIST
```

### `build_calib.py` — 生成校准数据

从真实数据集中提取样本，生成 NHWC float32 的 `.npy` 校准文件，供 `onnx2tf` INT8 量化使用。

- `data/calib_mnist_28x28x1_float32.npy`：从 MNIST test 集取 200 张，归一化到 0~1，形状 `[200, 28, 28, 1]`
- `data/calib_imagenet_224x224x3_float32.npy`：从 Imagenette val 集取 200 张，经 ImageNet 标准预处理（resize 256 → center crop 224 → 归一化 → mean/std 标准化），形状 `[200, 224, 224, 3]`

```bash
python build_calib.py                    # 生成全部校准数据
python build_calib.py --dataset mnist    # 仅生成 MNIST 校准数据
python build_calib.py --num-samples 100  # 指定样本数
```

### `convert.py` — 核心转换脚本

将 ONNX 模型转换为 TFLite 格式，支持 float32 和 INT8 两种模式。因 onnx2tf Python API 存在 segfault 问题，改用 subprocess 调用 CLI。

```bash
python convert.py --all                                  # 全部模型 float32
python convert.py --model mnist-12                       # 单模型 float32
python convert.py --all --quantize int8                  # 全部模型 INT8
python convert.py --model mobilenetv2-7 --quantize int8  # 单模型 INT8
```

转换流程对每个模型：
1. **ONNX 简化**：使用 `onnxsim` 去除冗余算子（`simplify_onnx()`）
2. **float32 转换**：调用 `onnx2tf -i <model> -o <output_dir>`（`convert_float32()`）
3. **INT8 量化转换**：调用 `onnx2tf -i <model> -o <output_dir> -oiqt -cind <input_name> <calib.npy> <mean> <std>`（`convert_int8()`）

### `validate.py` — 精度验证脚本

对比 ONNX、float32 TFLite、INT8 TFLite 三者的推理输出，验证转换与量化的精度。

```bash
python validate.py --all                    # 验证全部模型
python validate.py --model mnist-12         # 验证单模型
python validate.py --num-samples 20         # 指定验证样本数
```

验证内容：
- **ONNX 推理**：使用 onnxruntime，输入 NCHW 格式
- **float32 TFLite 推理**：使用 TFLite Interpreter，输入 NHWC 格式
- **INT8 TFLite 推理**：同上，但需正确处理 int8 量化的输入/输出
- 输出对比：最大/平均绝对误差、Top-1/Top-5 分类一致率、模型文件大小

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载模型与数据集
python download_models.py
python download_data.py

# 3. 生成校准数据
python build_calib.py

# 4. 转换（float32）
python convert.py --all

# 5. 转换（INT8 量化）
python convert.py --all --quantize int8

# 6. 验证
python validate.py --all --num-samples 5
```

---

## 关键代码定位

### ONNX 简化 — `convert.py:62`

```python
def simplify_onnx(input_path: str, output_path: str):
    """简化 ONNX 模型"""
    import onnxsim, onnx
    model = onnx.load(input_path)
    model_simp, _ = onnxsim.simplify(model)
    onnx.save(model_simp, output_path)
```

### float32 转换 — `convert.py:85`

```python
def convert_float32(name: str, config: dict):
    ...
    cmd = ['onnx2tf', '-i', simplified_path, '-o', output_dir]
    result = subprocess.run(cmd, ...)
```

核心是调用 `onnx2tf` CLI，将简化后的 ONNX 模型转为 TFLite。onnx2tf 内部会：ONNX → TensorFlow SavedModel → TFLite（float32/float16）。

### INT8 量化转换 — `convert.py:132`

```python
def convert_int8(name: str, config: dict):
    ...
    cmd = [
        'onnx2tf',
        '-i', simplified_path,
        '-o', output_dir,
        '-oiqt',                                                        # 开启整型量化输出
        '-cind', config['input_name'], calib_npy,                       # 校准数据
                  config['cind_mean'], config['cind_std'],              # 归一化参数
    ]
    result = subprocess.run(cmd, ...)
```

关键参数：
- **`-oiqt`**（`--output_integer_quantized_tflite`）：告诉 onnx2tf 输出整型量化模型
- **`-cind`**（`--calibration_input_op_name_data`）：提供校准数据，格式为 `<输入名> <.npy路径> <mean> <std>`

### INT8 推理时的量化/反量化 — `validate.py:89` / `validate.py:102`

INT8 模型的输入/输出张量是 int8 类型，推理时需要手动量化和反量化：

```python
def quantize_input(float_data, input_detail):
    """float32 → int8: quantized = round(float / scale + zero_point)"""
    scale = input_detail['quantization_parameters']['scales'][0]
    zp = input_detail['quantization_parameters']['zero_points'][0]
    return np.round(float_data / scale + zp).astype(input_detail['dtype'])

def dequantize_output(quant_data, output_detail):
    """int8 → float32: float = (quantized - zero_point) * scale"""
    scale = output_detail['quantization_parameters']['scales'][0]
    zp = output_detail['quantization_parameters']['zero_points'][0]
    return (quant_data.astype(np.float32) - zp) * scale
```

---

## INT8 量化原理

### 什么是量化

量化（Quantization）是将神经网络中 float32 的权重和激活值映射到低比特整数（如 int8）的过程，目的是减小模型体积、加速推理、降低内存带宽需求。

### 量化公式

对于 float32 值 `r` 与 int8 值 `q`，通过 **scale**（缩放因子）和 **zero_point**（零点偏移）建立映射：

```
量化：   q = round(r / scale + zero_point)
反量化： r = (q - zero_point) × scale
```

其中：
- `scale = (r_max - r_min) / (q_max - q_min)`，将 float 范围映射到 int8 范围 `[-128, 127]`
- `zero_point = round(q_min - r_min / scale)`，确保 float 的 0 能精确映射到某个 int8 值

### 两种量化方式

| 方式 | 说明 | 精度 |
|------|------|------|
| **训练后动态量化**（Dynamic Quantization） | 权重提前量化，激活值在推理时动态量化 | 较高 |
| **训练后静态量化**（Post-Training Static Quantization, PTQ） | 权重和激活值都提前量化，需要校准数据 | 中等，本项目使用 |

### 校准（Calibration）

静态量化需要知道激活值的数值范围（min/max）才能计算 scale 和 zero_point。校准过程：

1. 准备一批**有代表性的真实输入样本**（通常 100~500 张）
2. 用 float32 模型对校准数据做前向推理，**统计每层激活值的分布**
3. 根据 min/max 计算 scale 和 zero_point

校准数据的质量直接影响量化精度——如果校准数据的分布不能代表真实输入，量化后的模型精度会显著下降。本项目使用 MNIST test 集和 Imagenette val 集的真实图片作为校准数据，而非随机噪声。

### INT8 量化对精度的影响

量化不可避免地引入误差，但对分类网络而言，输出是 logits 而非概率值，绝对数值的误差不等于分类结果的改变。通常：

- **权重量化**：误差较小，权重分布相对集中
- **激活量化**：误差较大，激活值分布因输入不同而变化剧烈，这是量化精度损失的主要来源

本项目实测结果：

| 模型 | float32 Top-1 | INT8 Top-1 | 体积压缩 |
|------|--------------|------------|---------|
| MNIST-12 | 100% | 100% | 2.6× |
| SqueezeNet1.1-7 | 100% | 80% | 3.6× |
| MobileNetV2-7 | 100% | 80% | 3.5× |
| ResNet18-v1-7 | 100% | 100% | 3.9× |

MNIST 结构简单、激活值分布稳定，量化几乎无损；SqueezeNet/MobileNetV2 激活值分布较分散，Top-1 有轻微下降；ResNet18 残差结构对量化误差更鲁棒。如需进一步提升量化精度，可考虑**量化感知训练（QAT）**。

---

## 验证结果

使用 5 张真实图片验证（`python validate.py --all --num-samples 5`）：

### float32 TFLite vs ONNX

| 模型 | 最大绝对误差 | Top-1 一致率 | Top-5 一致率 |
|------|-------------|-------------|-------------|
| MNIST-12 | 0.000004 | 100% | 100% |
| SqueezeNet1.1-7 | 0.000012 | 100% | 100% |
| MobileNetV2-7 | 0.000040 | 100% | 100% |
| ResNet18-v1-7 | 0.000019 | 100% | 100% |

### INT8 TFLite vs ONNX

| 模型 | 最大绝对误差 | Top-1 一致率 | Top-5 一致率 |
|------|-------------|-------------|-------------|
| MNIST-12 | 0.18 | 100% | 100% |
| SqueezeNet1.1-7 | 3.77 | 80% | 40% |
| MobileNetV2-7 | 14.40 | 80% | 20% |
| ResNet18-v1-7 | 12.86 | 100% | 80% |

### 模型体积对比

| 模型 | ONNX | float32 TFLite | INT8 TFLite | 压缩比 |
|------|------|---------------|-------------|--------|
| MNIST-12 | 25.5 KB | 26.1 KB | 9.9 KB | 2.6× |
| SqueezeNet1.1-7 | 4.7 MB | 4.7 MB | 1.3 MB | 3.6× |
| MobileNetV2-7 | 13.6 MB | 13.3 MB | 3.8 MB | 3.5× |
| ResNet18-v1-7 | 44.7 MB | 44.6 MB | 11.3 MB | 3.9× |

---

## 参考资料

### 工具与仓库

- [onnx2tf](https://github.com/PINTO0309/onnx2tf) — ONNX → TFLite 转换工具，支持 INT8 量化
- [ONNX Model Zoo on Hugging Face](https://huggingface.co/onnxmodelzoo) — 本项目使用的公开 ONNX 模型来源
- [TensorFlow Lite](https://www.tensorflow.org/lite) — TFLite 官方文档
- [AI Edge LiteRT](https://ai.google.dev/edge/litert) — TFLite 的后续版本（原 TFLite 迁移目标）

### 量化原理

- [TensorFlow 官方：Post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization) — TFLite 训练后量化指南，详述动态量化、静态量化和 Full Integer 量化的区别
- [TensorFlow 官方：Quantization specification](https://www.tensorflow.org/lite/performance/quantization_spec) — TFLite 量化规范，定义 scale/zero_point 的计算方式
- [ONNX Runtime 量化文档](https://onnxruntime.ai/docs/performance/quantization.html) — ONNX 侧的量化方法说明，PTQ 与 QAT 的对比
- [MIT 6.S191：Quantization](https://intellabs.github.io/distiller/quantization.html) — Intel Distiller 项目文档，深入讲解量化的数学原理（对称/非对称量化、per-channel vs per-tensor）

### 数据集

- [Imagenette](https://github.com/fastai/imagenette) — ImageNet 10 类子集，免注册下载
- [MNIST 公开镜像](https://ossci-datasets.s3.amazonaws.com/mnist/) — AWS S3 镜像，无需注册
- [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k) — 完整 ImageNet 数据集（需注册同意条款）

### 博客与教程

- [PINTO0309：onnx2tf 使用说明](https://github.com/PINTO0309/onnx2tf#usage) — onnx2tf 作者的详细参数说明，包括 `-oiqt` 和 `-cind` 的用法
- [TensorFlow Blog：Introducing the Model Optimization Toolkit](https://blog.tensorflow.org/2019/06/tensorflow-model-optimization-toolkit_10.html) — TF 官方博客，介绍模型优化工具包及量化技术
- [PyTorch 官方：Quantization Tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html) — PyTorch 静态量化教程，量化原理通用可参考
- [Qualcomm：Quantization for Neural Networks](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk) — 高通 AI 引擎文档中的量化章节，涵盖 PTQ 与 QAT 的工程实践
