# 公开 ONNX 模型与正式数据集下载指南

本文档整理了一批可用于验证本项目 ONNX → TFLite 转换功能的**公开 ONNX 模型**，以及配套的**正式标注数据集**的获取方式。这些数据集除了用于验证精度，**后续还将用作 INT8 量化的校准数据（calibration dataset）**，因此全部采用真实样本来源，不使用随机数据。

---

## 一、重要背景说明

ONNX 官方 Model Zoo 仓库（`github.com/onnx/models`）已于 **2025 年 7 月 1 日** 停止通过 Git LFS 提供模型下载。所有模型现已迁移至 Hugging Face：

- 新地址：<https://huggingface.co/onnxmodelzoo>

> **注意**：迁移后仓库中**只保留了 `.onnx` 模型文件和 `README.md`**，原先随模型打包的 `test_data_set_0/`（`input_0.pb` + 参考输出 `output_0.pb`）未一并迁移。因此模型与数据集需分开下载，数据集来源见下文【三】【四】。

---

## 二、推荐的 ONNX 模型

以下模型均来自官方 `onnxmodelzoo`，许可证为 **Apache-2.0**，经典 CNN 结构、算子支持完整，对 `onnx2tf` 转换器友好，且都有明确对应的公开数据集，便于做精度验证与量化校准。

| 模型 | Hugging Face 仓库 | 大小 | 输入形状 (NCHW) | 输出 | 训练数据集 |
|------|------|------|------|------|------|
| **MNIST-12** | `onnxmodelzoo/mnist-12` | ~26 KB | `[1, 1, 28, 28]` | `[1, 10]` | MNIST |
| **SqueezeNet1.1-7** | `onnxmodelzoo/squeezenet1.1-7` | ~5 MB | `[1, 3, 224, 224]` | `[1, 1000]` | ImageNet-1000 |
| **MobileNetV2-7** | `onnxmodelzoo/mobilenetv2-7` | ~14 MB | `[1, 3, 224, 224]` | `[1, 1000]` | ImageNet-1000 |
| **ResNet18-v1-7** | `onnxmodelzoo/resnet18-v1-7` | ~45 MB | `[1, 3, 224, 224]` | `[1, 1000]` | ImageNet-1000 |

### 2.1 直接下载（推荐，无需 git-lfs）

```bash
mkdir -p models

wget https://huggingface.co/onnxmodelzoo/mnist-12/resolve/main/mnist-12.onnx -O models/mnist-12.onnx
wget https://huggingface.co/onnxmodelzoo/squeezenet1.1-7/resolve/main/squeezenet1.1-7.onnx -O models/squeezenet1.1-7.onnx
wget https://huggingface.co/onnxmodelzoo/mobilenetv2-7/resolve/main/mobilenetv2-7.onnx -O models/mobilenetv2-7.onnx
wget https://huggingface.co/onnxmodelzoo/resnet18-v1-7/resolve/main/resnet18-v1-7.onnx -O models/resnet18-v1-7.onnx
```

### 2.2 huggingface-cli / 国内镜像

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download onnxmodelzoo/mobilenetv2-7 mobilenetv2-7.onnx --local-dir models

# 国内网络可用 hf-mirror 镜像
export HF_ENDPOINT=https://hf-mirror.com
wget https://hf-mirror.com/onnxmodelzoo/mnist-12/resolve/main/mnist-12.onnx -O models/mnist-12.onnx
```

下载 URL 统一规律：`https://huggingface.co/onnxmodelzoo/<仓库名>/resolve/main/<文件名>.onnx`，完整列表见 <https://huggingface.co/onnxmodelzoo>。

---

## 三、正式数据集：MNIST（配 MNIST-12）

MNIST 为手写数字 0~9，单通道 28×28。以下来源均为真实标注数据，可同时用于精度验证和量化校准。

| 来源 | 获取方式 | 是否需注册 |
|------|------|------|
| **AWS 公开镜像（原始 idx 文件）** | 见下方命令 | 否 |
| **TensorFlow / Keras 内置** | `tf.keras.datasets.mnist.load_data()` | 否 |
| **PyTorch / torchvision** | `torchvision.datasets.MNIST(root='./data', download=True)` | 否 |
| **Hugging Face Datasets** | `load_dataset("ylecun/mnist")`（需 `pip install datasets`） | 否 |

AWS 镜像直接下载（官方原站点不稳定，推荐此源）：

```bash
mkdir -p data/mnist
BASE=https://ossci-datasets.s3.amazonaws.com/mnist
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
         t10k-images-idx3-ubyte.gz  t10k-labels-idx1-ubyte.gz; do
  wget "$BASE/$f" -O "data/mnist/$f"
done
```

**预处理**：像素 0~255 → 归一化到 0~1（部分模型按 `mean=0.1307, std=0.3081` 标准化，以模型 README 为准）；输入 ONNX 为 NCHW `[N,1,28,28]`，转 TFLite 推理时 transpose 为 NHWC `[N,28,28,1]`。

---

## 四、正式数据集：ImageNet（配 SqueezeNet / MobileNetV2 / ResNet18）

这三个模型均在 **ImageNet-1000（ILSVRC 2012）** 上训练，输入 224×224 RGB。根据是否需要完整验证集，有以下几种选择。

### 4.1 完整 ImageNet-1000 验证集（最正式，需注册）

50000 张验证图、1000 类，是量化校准与精度评测的标准数据源。需在 Hugging Face 登录并同意条款后下载：

- 数据集主页：<https://huggingface.co/datasets/imagenet-1k>
- 数据目录（含 `val_images.tar.gz`）：<https://huggingface.co/datasets/imagenet-1k/tree/main/data>

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login   # 需先在网页同意数据集使用条款

huggingface-cli download imagenet-1k --repo-type dataset \
  --include "data/val_images.tar.gz" --local-dir data/imagenet
```

> 官方原站 <https://www.image-net.org/> 同样提供下载，但同样需要注册申请。

### 4.2 Imagenette —— ImageNet 真实子集（推荐，免注册）

Imagenette 是 fastai 从 ImageNet 中抽取的 **10 个类别的真实图片子集**（tench、English springer、cassette player、chain saw、church、French horn、garbage truck、gas pump、golf ball、parachute），图片本身就是 ImageNet 原图，非常适合做**量化校准**和快速精度验证，且**无需注册**直接下载。

```bash
mkdir -p data
# 全尺寸（约 1.5 GB）
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz -O data/imagenette2.tgz
# 320px 短边版（推荐用于校准，体积适中）
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz -O data/imagenette2-320.tgz
# 160px 短边版（最小，约 100 MB）
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz -O data/imagenette2-160.tgz

tar -xzf data/imagenette2-320.tgz -C data
# 解压后目录结构：data/imagenette2-320/{train,val}/<类别wnid>/*.JPEG
```

Hugging Face Datasets 形式（同一数据集）：

```python
from datasets import load_dataset
ds = load_dataset("frgfm/imagenette", "320px", split="validation")
```

> **校准注意**：Imagenette 只含 10 个类别。若仅做转换功能验证和量化校准（校准只需"代表性输入分布"，不依赖标签覆盖全 1000 类），Imagenette 足够；若要复现官方 top-1/top-5 精度指标，应使用 4.1 的完整验证集。

### 4.3 其他可选子集（免注册）

| 数据集 | 说明 | 地址 |
|------|------|------|
| **ImageNet-100** | 100 类子集，train 12.7 万 / val 5 千 | <https://huggingface.co/datasets/clane9/imagenet-100> |
| **类别标签映射** | `imagenet_classes.txt`（1000 类索引→名称），用于解读输出 | 各框架示例仓库（如 PyTorch Hub 示例）均有 |

### 4.4 ImageNet 标准预处理

ImageNet 系模型推理 / 校准前需统一完成：

1. Resize 短边到 256，再 CenterCrop 到 224×224；
2. 像素归一化到 0~1；
3. 标准化：`mean = [0.485, 0.456, 0.406]`，`std = [0.229, 0.224, 0.225]`；
4. ONNX 输入为 NCHW `[N,3,224,224]`；转 TFLite 推理 / 校准时 transpose 为 NHWC `[N,224,224,3]`。

> 具体参数以各模型 Hugging Face 仓库 `README.md` 为准。

---

## 五、用于 INT8 量化校准（calibration）的数据准备

量化校准需要一批**真实、有代表性的输入样本**让量化器统计各层激活值的动态范围。要点：

- **样本数量**：通常 100~500 张即可，覆盖典型输入分布即可，不必用全量数据。
- **预处理一致**：校准样本必须经过与推理**完全相同**的预处理（见三、四节）。
- **数据布局**：本项目转 TFLite 走 `onnx2tf`，其校准数据为 **NHWC、float32** 的 `.npy`，形状 `[N, H, W, C]`。

### 5.1 制作校准用 .npy（ImageNet 模型示例）

```python
# build_calib.py —— generate an NHWC float32 .npy from Imagenette for INT8 calibration
import os, glob, numpy as np
from PIL import Image

SRC = "data/imagenette2-320/val"   # real ImageNet-subset images
N   = 200                          # number of calibration samples
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(path):
    img = Image.open(path).convert("RGB")
    # resize shortest side to 256, then center crop 224x224
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD               # HWC, normalized
    return x

files = glob.glob(os.path.join(SRC, "*", "*.JPEG"))[:N]
data = np.stack([preprocess(f) for f in files], axis=0)  # [N, 224, 224, 3] NHWC
np.save("calib_imagenet_224x224x3_float32.npy", data)
print("saved:", data.shape, data.dtype)
```

### 5.2 在 onnx2tf 中使用校准数据

`onnx2tf` 通过 INT8 量化相关选项读取上面的 `.npy` 进行校准（生成 INT8 TFLite）。常用方式是用 `-oiqt` 输出整型量化模型，并用 `-cind` 指定输入名与校准数据路径：

```bash
# 形如：-cind "<input_name>" "<calib.npy>" "<mean>" "<std>"
onnx2tf -i models/mobilenetv2-7.onnx -o output \
  -oiqt \
  -cind "input" "calib_imagenet_224x224x3_float32.npy" "[[[[0.485,0.456,0.406]]]]" "[[[[0.229,0.224,0.225]]]]"
```

> 具体参数名（输入名、是否在 `.npy` 内已含归一化等）以 `onnx2tf -h` 和其 README 为准：<https://github.com/PINTO0309/onnx2tf>。本项目 `convert.py` 当前用一个随机 `.npy`（`calibration_image_sample_data_20x128x128x3_float32.npy`）仅为绕过 onnx2tf 的下载逻辑，**正式量化时应替换为上面用真实数据生成的校准文件**。

---

## 六、下载链接速查

### 模型

| 模型 | URL |
|------|------|
| MNIST-12 | `https://huggingface.co/onnxmodelzoo/mnist-12/resolve/main/mnist-12.onnx` |
| SqueezeNet1.1-7 | `https://huggingface.co/onnxmodelzoo/squeezenet1.1-7/resolve/main/squeezenet1.1-7.onnx` |
| MobileNetV2-7 | `https://huggingface.co/onnxmodelzoo/mobilenetv2-7/resolve/main/mobilenetv2-7.onnx` |
| ResNet18-v1-7 | `https://huggingface.co/onnxmodelzoo/resnet18-v1-7/resolve/main/resnet18-v1-7.onnx` |

### 数据集

| 数据集 | URL | 注册 |
|------|------|------|
| MNIST（AWS 镜像） | `https://ossci-datasets.s3.amazonaws.com/mnist/` | 否 |
| ImageNet-1k 完整验证集 | `https://huggingface.co/datasets/imagenet-1k/tree/main/data` | 是 |
| Imagenette 全尺寸 | `https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz` | 否 |
| Imagenette 320px | `https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz` | 否 |
| Imagenette 160px | `https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz` | 否 |
| ImageNet-100 | `https://huggingface.co/datasets/clane9/imagenet-100` | 否 |

---

## 参考资料

- ONNX Model Zoo on Hugging Face：<https://huggingface.co/onnxmodelzoo>
- onnx2tf 转换工具：<https://github.com/PINTO0309/onnx2tf>
- ImageNet-1k 数据集：<https://huggingface.co/datasets/imagenet-1k>
- Imagenette 数据集：<https://github.com/fastai/imagenette>
- MNIST 公开镜像：<https://ossci-datasets.s3.amazonaws.com/mnist/>
- ImageNet 官网：<https://www.image-net.org/>
- HF 国内镜像：<https://hf-mirror.com>
