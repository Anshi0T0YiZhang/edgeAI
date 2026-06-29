# ONNX 与 TFLite 模型格式学习笔记

本项目介绍两种主流的模型封装格式：ONNX 和 TFLite，并演示如何创建和转换模型。

## 运行环境

| 项目 | 信息 |
|------|------|
| **操作系统** | Ubuntu 24.04.4 LTS (WSL2) |
| **WSL 内核** | Linux 6.6.87.2-microsoft-standard-WSL2 |
| **GPU** | NVIDIA GeForce RTX 5060 Laptop GPU |
| **NVIDIA 驱动** | 581.29 |
| **CUDA 版本** | 13.0 |
| **Python** | 3.11.8 |

> 注：本项目在 CPU 模式下运行，TensorFlow 未启用 GPU 加速。



## 一、ONNX (Open Neural Network Exchange)

### 1.1 什么是 ONNX？

ONNX（Open Neural Network Exchange）是一种开放的神经网络交换格式，由 Facebook（Meta）和 Microsoft 于 2017 年联合开发并开源。它的核心目标是实现不同深度学习框架之间的模型互操作性。

### 1.2 ONNX 的核心特点

| 特性 | 说明 |
|------|------|
| **跨框架兼容** | 支持 PyTorch、TensorFlow、MXNet、CNTK 等主流框架导出 |
| **标准化算子** | 定义了统一的算子集（Operators），确保模型行为一致 |
| **Opset 版本** | 通过 opset_version 管理算子版本，保证向后兼容 |
| **计算图表示** | 使用有向无环图（DAG）表示模型结构 |

### 1.3 ONNX 文件结构

ONNX 使用 Protocol Buffers 进行序列化，主要包含以下组件：

```
ModelProto
├── ir_version: IR 版本号
├── opset_import: 使用的算子集版本
├── producer_name/version: 生成工具信息
├── graph: 计算图 (GraphProto)
│   ├── name: 图名称
│   ├── input: 输入张量列表 (ValueInfoProto)
│   ├── output: 输出张量列表 (ValueInfoProto)
│   ├── initializer: 初始化参数（权重）(TensorProto)
│   └── node: 计算节点列表 (NodeProto)
│       ├── op_type: 算子类型（如 Conv, MatMul, Relu）
│       ├── input: 输入张量名称
│       ├── output: 输出张量名称
│       └── attribute: 算子属性
└── metadata_props: 元数据
```

### 1.4 ONNX 使用场景

- **模型迁移**：将 PyTorch 训练的模型迁移到其他推理框架
- **跨平台部署**：ONNX Runtime 支持多种硬件加速（CUDA、TensorRT、OpenVINO）
- **模型优化**：通过 ONNX 作为中间格式进行图优化
- **模型验证**：标准化的格式便于模型正确性验证

---

## 二、TFLite (TensorFlow Lite)

### 2.1 什么是 TFLite？

TensorFlow Lite 是 Google 为移动端和嵌入式设备优化的轻量级推理框架。TFLite 模型（`.tflite` 文件）是专门为边缘计算场景设计的模型格式。

### 2.2 TFLite 的核心特点

| 特性 | 说明 |
|------|------|
| **轻量级** | 模型体积小，适合存储空间有限的设备 |
| **快速推理** | 针对移动 CPU/GPU/DSP 优化 |
| **量化支持** | 支持 INT8、Float16 量化，大幅降低模型大小和推理延迟 |
| **跨平台** | 支持 Android、iOS、Linux、MCU 等多种平台 |
| **硬件加速** | 支持 NNAPI、Core ML、GPU Delegate 等 |

### 2.3 TFLite 文件结构

TFLite 使用 FlatBuffers 进行序列化，主要结构如下：

```
Model
├── version: 模型版本
├── operator_codes: 算子代码表
├── subgraphs: 子图列表
│   ├── tensors: 张量列表（包含权重数据）
│   ├── inputs: 输入张量索引
│   ├── outputs: 输出张量索引
│   └── operators: 算子列表
│       ├── opcode_index: 算子代码索引
│       ├── inputs: 输入张量索引
│       └── outputs: 输出张量索引
├── description: 模型描述
└── buffers: 数据缓冲区（存储权重等）
```

### 2.4 TFLite 量化类型

| 量化类型 | 精度 | 模型大小 | 适用场景 |
|---------|------|---------|---------|
| Float32 | 全精度 | 基准 | 精度要求高 |
| Float16 | 半精度 | 减半 | GPU 加速 |
| INT8 | 8位整数 | 1/4 | 边缘设备、MCU |

### 2.5 TFLite 使用场景

- **移动应用**：Android/iOS 应用中的 AI 功能
- **物联网设备**：Raspberry Pi、Jetson Nano 等边缘设备
- **微控制器**：TinyML 场景，如 STM32、ESP32
- **实时推理**：对延迟敏感的应用

---

## 三、ONNX vs TFLite 对比

| 对比维度 | ONNX | TFLite |
|---------|------|--------|
| **开发方** | Meta + Microsoft | Google |
| **序列化格式** | Protocol Buffers | FlatBuffers |
| **文件扩展名** | `.onnx` | `.tflite` |
| **数据格式** | NCHW (Batch, Channel, Height, Width) | NHWC (Batch, Height, Width, Channel) |
| **主要用途** | 模型交换、跨框架迁移 | 移动端/边缘部署 |
| **推理引擎** | ONNX Runtime | TFLite Interpreter |
| **硬件支持** | CUDA、TensorRT、OpenVINO、ROCm | NNAPI、Core ML、GPU Delegate、Edge TPU |
| **量化支持** | 有（QDQ） | 原生支持多种量化方案 |
| **模型大小** | 较大 | 较小（优化后） |
| **生态系统** | 跨框架通用 | TensorFlow 生态专属 |

---

## 四、转换流程

### 4.1 为什么需要转换？

1. **训练框架与部署环境不匹配**：使用 PyTorch 训练，部署到移动端需要 TFLite
2. **模型优化需求**：TFLite 提供更丰富的量化选项
3. **硬件兼容性**：某些硬件仅支持特定格式

### 4.2 转换工具链

本项目采用标准的 ONNX → TFLite 转换流程：

```
创建 ONNX 模型 → ONNX-SIMPLIFIER 简化 → onnx2tf 转换 → TFLite
```

**各环节说明：**
- **创建 ONNX**：使用 ONNX API 定义计算图结构
- **简化模型**：`onnx-simplifier` 优化计算图，合并冗余节点
- **转换为 TFLite**：`onnx2tf` 将 ONNX 算子映射到 TFLite 格式

> 注：由于 onnx2tf 内部加载测试数据的兼容性问题，脚本会自动创建本地缓存文件来绕过。

### 4.3 转换命令

#### 安装依赖
```bash
pip install -r requirements.txt
```

#### 转换模型
```bash
# 创建示例 ONNX 模型并转换为 TFLite
python convert.py

# 转换指定的 ONNX 模型
python convert.py -i your_model.onnx

# 指定输出目录
python convert.py -i your_model.onnx -o output_dir
```

#### 转换结果
执行后会生成以下文件：
```
models/
├── sample.onnx                # 原始 ONNX 模型
├── sample_sim.onnx            # 简化后的 ONNX 模型
├── sample_sim_float32.tflite  # TFLite Float32 模型
└── sample_sim_float16.tflite  # TFLite Float16 量化模型
```

#### 测试模型
```bash
# 测试推理
python test_model.py

# 查看 ONNX 模型结构
python test_model.py --onnx models/sample_sim.onnx --inspect

# 查看 TFLite 模型结构
python test_model.py --tflite models/sample_sim_float32.tflite --inspect

# 性能基准测试 (100次)
python test_model.py --benchmark --runs 100
```

#### ONNX 简化（可选）
```bash
# 使用 onnx-simplifier 简化模型
onnxsim input.onnx simplified.onnx
```

#### 使用 onnx2tf 转换（需要 numpy < 2.0）
```bash
# 转换 ONNX 到 TFLite
onnx2tf -i input.onnx -o output_folder

# 使用自定义输入数据
onnx2tf -i input.onnx -o output_folder -cind input_name input.npy
```

---

## 五、项目文件说明

```
edgeAI/
├── requirements.txt       # Python 依赖列表
├── README.md              # 本学习文档
├── convert.py             # ONNX 到 TFLite 转换脚本
├── test_model.py          # 模型推理测试脚本
└── models/                # 模型存放目录
    ├── sample.onnx                # 原始 ONNX 模型 (21.8 KB)
    ├── sample_sim.onnx            # 简化后 ONNX 模型 (22.0 KB)
    ├── sample_sim_float32.tflite  # TFLite Float32 (23.2 KB)
    └── sample_sim_float16.tflite  # TFLite Float16 (13.3 KB)
```

### 模型信息

| 模型 | 输入格式 | 输入形状 | 输出形状 |
|------|---------|---------|---------|
| `sample.onnx` | NCHW | [1, 3, 64, 64] | [1, 10] |
| `sample_sim.onnx` | NCHW | [1, 3, 64, 64] | [1, 10] |
| `sample_sim_float32.tflite` | NHWC | [1, 64, 64, 3] | [1, 10] |
| `sample_sim_float16.tflite` | NHWC | [1, 64, 64, 3] | [1, 10] |

---

## 六、常见问题

### Q1: onnx2tf 转换时报错无法加载测试数据？
这是 onnx2tf 内部的问题。本脚本会自动创建本地缓存文件来绕过。

### Q2: ONNX 和 TFLite 输入格式不同？
- **ONNX** 使用 NCHW 格式：`[Batch, Channel, Height, Width]`
- **TFLite** 使用 NHWC 格式：`[Batch, Height, Width, Channel]`

onnx2tf 转换时会自动处理数据格式的转换。

### Q3: 如何查看模型结构？
推荐使用 [Netron](https://netron.app/) 可视化工具，支持 ONNX 和 TFLite 格式。

---

## 七、参考资料

- [ONNX 官方网站](https://onnx.ai/)
- [ONNX GitHub](https://github.com/onnx/onnx)
- [TensorFlow Lite 官方文档](https://www.tensorflow.org/lite)
- [onnx2tf GitHub](https://github.com/PINTO0309/onnx2tf)
- [onnx-simplifier GitHub](https://github.com/daquexian/onnx-simplifier)
- [Netron 模型可视化](https://netron.app/)