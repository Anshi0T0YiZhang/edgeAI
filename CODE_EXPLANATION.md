# 代码实现详解

本文档详细解释 ONNX 到 TFLite 转换过程中各部分代码的功能、数据结构和转换流程。

---

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    convert.py 转换脚本                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  创建 ONNX   │ →  │  简化 ONNX   │ →  │  onnx2tf 转换 │   │
│  │    模型      │    │    模型      │    │  为 TFLite   │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                   │                    │          │
│         ↓                   ↓                    ↓          │
│  sample.onnx          sample_sim.onnx     sample_sim*.tflite│
│  (NCHW格式)           (优化后结构)       (NHWC格式+量化)   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、ONNX 模型创建

### 2.1 数据格式（NCHW）

ONNX 标准使用 **NCHW** 数据布局：

```
┌─────────────────────────────────────────────────────────────┐
│                        输入张量 [1, 3, 64, 64]                  │
├─────────────┬───────────┬───────────┬───────────┬─────────────┤
│  N = 1      │  C = 3    │  H = 64   │  W = 64   │   dtype    │
│  (批次大小)  │  (通道数)  │  (高度)   │  (宽度)   │  float32   │
└─────────────┴───────────┴───────────┴───────────┴─────────────┘

内存布局 (C dimension 最快变化):
Channel 0: [0:64, 0:64]  →  R通道 (红色)
Channel 1: [0:64, 0:64]  →  G通道 (绿色)
Channel 2: [0:64, 0:64]  →  B通道 (蓝色)
```

### 2.2 模型结构定义

```python
# 卷积层权重 - ONNX 格式为 [C_out, C_in, H_k, W_k]
conv1_w: [16, 3, 3, 3]   # 16个输出通道, 3个输入通道, 3x3卷积核
conv1_b: [16]             # 16个偏置值

conv2_w: [32, 16, 3, 3]  # 32个输出通道, 16个输入通道, 3x3卷积核
conv2_b: [32]             # 32个偏置值

# 全连接层权重
fc_w: [32, 10]            # 32个输入特征, 10个输出类别
fc_b: [10]                # 10个偏置值
```

### 2.3 计算图节点

```python
┌─────────────────────────────────────────────────────────────┐
│                     ONNX 计算图流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  input [1,3,64,64]                                           │
│       │                                                      │
│       ▼                                                      │
│  Conv [1,16,64,64]   ← 卷积 + 偏置                          │
│       │                                                      │
│       ▼                                                      │
│  ReLU  [1,16,64,64]   ← 激活函数                             │
│       │                                                      │
│       ▼                                                      │
│  MaxPool [1,16,32,32]  ← 2x2池化, 步长2                      │
│       │                                                      │
│       ▼                                                      │
│  Conv [1,32,32,32]   ← 卷积 + 偏置                          │
│       │                                                      │
│       ▼                                                      │
│  ReLU  [1,32,32,32]   ← 激活函数                             │
│       │                                                      │
│       ▼                                                      │
│  GlobalAvgPool [1,32,1,1] ← 全局平均池化                    │
│       │                                                      │
│       ▼                                                      │
│  Reshape [1,32]       ← 展平为 1D                           │
│       │                                                      │
│       ▼                                                      │
│  MatMul [1,10]         ← 全连接层 (x @ W + B)                │
│       │                                                      │
│       ▼                                                      │
│  output [1,10]        ← 最终输出 (10类分类概率)              │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、ONNX 简化

### 3.1 简化目的

`onnx-simplifier` 优化计算图，主要操作：

| 操作类型 | 说明 | 示例 |
|---------|------|------|
| **常量折叠** | 预计算常量表达式 | `Add(2, 3)` → `5` |
| **死节点消除** | 移除未使用的节点 | 未连接的节点 |
| **算子融合** | 合并连续操作 | `Conv+BN+ReLU` → 融合节点 |
| **冗余消除** | 移除恒等变换 | `Identity` 节点 |

### 3.2 简化前后对比

```python
# 简化前
节点数: 10
模型大小: 21.8 KB

# 简化后
节点数: 10 (本例已简化，无明显变化)
模型大小: 22.0 KB (元数据增加)
```

---

## 四、ONNX 到 TFLite 转换

### 4.1 数据格式转换（NCHW → NHWC）

onnx2tf 自动转换数据布局：

```
ONNX 格式 (NCHW):
[B, C, H, W] = [1, 3, 64, 64]

    内存顺序: C fastest changing
    ↓
    转换操作: transpose(0, 2, 3, 1)
    ↓
TFLite 格式 (NHWC):
[B, H, W, C] = [1, 64, 64, 3]

    内存顺序: W fastest changing (TensorFlow 优化)
```

### 4.2 算子映射

| ONNX 算子 | TFLite 算子 | 说明 |
|----------|------------|------|
| `Conv` | `CONV_2D` | 卷积操作 |
| `Relu` | `RELU` | ReLU 激活 |
| `MaxPool` | `MAX_POOL_2D` | 最大池化 |
| `GlobalAveragePool` | `MEAN` | 归约求平均 |
| `Reshape` | `RESHAPE` | 张量重塑 |
| `MatMul` | `FULLY_CONNECTED` | 全连接层 |
| `Add` | `ADD` | 逐元素加法 |

### 4.3 量化

onnx2tf 默认生成两种量化版本：

#### Float16 量化
```
权重: float32 → float16
激活: float32 → float16
优势: 模型大小减半，GPU 加速更快
```

#### Float32 (原始)
```
权重: float32 (保持不变)
激活: float32 (保持不变)
优势: 精度最高
```

---

## 五、输出文件结构

### 5.1 文件列表

```
models/
├── sample.onnx
│   ├── IR版本: 7
│   ├── Opset: 11
│   ├── 输入: [1, 3, 64, 64] (NCHW, float32)
│   ├── 输出: [1, 10] (float32)
│   └── 节点: Conv, Relu, MaxPool, Conv, Relu, GAP, Reshape, MatMul, Add
│
├── sample_sim.onnx
│   ├── 优化后的 ONNX
│   ├── 相同的输入输出接口
│   └── 计算图经过优化
│
├── sample_sim_float32.tflite
│   ├── 输入: [1, 64, 64, 3] (NHWC, float32)
│   ├── 输出: [1, 10] (float32)
│   ├── 大小: 23.2 KB
│   └── 算子: CONV_2D, RELU, MAX_POOL_2D, MEAN, RESHAPE, FULLY_CONNECTED, ADD
│
└── sample_sim_float16.tflite
    ├── 输入: [1, 64, 64, 3] (NHWC, float16)
    ├── 输出: [1, 10] (float16)
    ├── 大小: 13.3 KB (减少 43%)
    └── 算子: 同 Float32 版本
```

### 5.2 模型大小对比

| 模型 | 格式 | 精度 | 大小 | 相对比例 |
|------|------|------|------|----------|
| sample.onnx | ONNX | Float32 | 21.8 KB | 100% |
| sample_sim.onnx | ONNX | Float32 | 22.0 KB | 101% |
| sample_sim_float32.tflite | TFLite | Float32 | 23.2 KB | 106% |
| sample_sim_float16.tflite | TFLite | Float16 | 13.3 KB | 61% |

---

## 六、代码函数详解

### 6.1 create_sample_onnx()

```python
def create_sample_onnx(output_path: str):
    """创建 ONNX 模型
    
    步骤:
    1. 定义层权重 (使用 numpy_helper.from_array)
    2. 创建计算节点 (helper.make_node)
    3. 定义输入输出张量 (make_tensor_value_info)
    4. 构建计算图 (helper.make_graph)
    5. 创建模型 (helper.make_model)
    6. 验证并保存 (onnx.checker, onnx.save)
    """
```

### 6.2 simplify_onnx()

```python
def simplify_onnx(input_path: str, output_path: str):
    """简化 ONNX 模型
    
    步骤:
    1. 加载原始模型 (onnx.load)
    2. 运行简化器 (onnxsim.simplify)
    3. 保存简化后模型
    """
```

### 6.3 onnx_to_tflite()

```python
def onnx_to_tflite(onnx_path: str, output_dir: str):
    """ONNX 到 TFLite 转换
    
    步骤:
    1. 创建本地缓存文件 (绕过 onnx2tf 的下载问题)
    2. 调用 onnx2tf.convert()
       - 加载 ONNX 模型
       - 算子映射 (ONNX → TensorFlow)
       - 生成 SavedModel
       - 转换为 TFLite (Float32 + Float16)
    3. 返回生成的 TFLite 文件列表
    """
```

---

## 七、数据流示例

### 7.1 输入数据流转

```python
# 1. 生成随机输入 (NCHW格式)
input_data = np.random.rand(1, 3, 64, 64).astype(np.float32)

# 2. ONNX Runtime 推理
session = ort.InferenceSession("sample_sim.onnx")
output = session.run(None, {"input": input_data})
# 输出形状: (1, 10)

# 3. 转换为 TFLite 输入格式 (NHWC)
input_nhwc = np.transpose(input_data, (0, 2, 3, 1))
# 形状: (1, 64, 64, 3)

# 4. TFLite 推理
interpreter = tf.lite.Interpreter("sample_sim_float32.tflite")
interpreter.set_tensor(input_idx, input_nhwc)
interpreter.invoke()
output = interpreter.get_tensor(output_idx)
# 输出形状: (1, 10)
```

---

## 八、关键知识点

### 8.1 NCHW vs NHWC

| 特性 | NCHW (ONNX) | NHWC (TFLite) |
|------|-------------|---------------|
| **内存布局** | 通道优先 | 空间优先 |
| **优势** | CPU 友好 | GPU/ARM 优化 |
| **转换** | transpose(0,2,3,1) | transpose(0,3,1,2) |
| **常用库** | ONNX, PyTorch | TensorFlow, TFLite |

### 8.2 序列化格式

| 格式 | 库 | 特点 |
|------|------|------|
| **Protocol Buffers** | ONNX | Google 开源，Schema 定义清晰 |
| **FlatBuffers** | TFLite | Google 开源，零拷贝，适合移动端 |

### 8.3 量化类型

| 类型 | 字节数 | 范围 | 适用场景 |
|------|--------|------|----------|
| Float32 | 4 | ±3.4×10³⁸ | 精度要求高 |
| Float16 | 2 | ±6.5×10⁴ | GPU 加速 |
| INT8 | 1 | -128~127 | 边缘设备 |

---

## 九、验证转换正确性

### 9.1 输出对比

```bash
# 测试两种格式的输出一致性
python test_model.py --compare --onnx models/sample_sim.onnx --tflite models/sample_sim_float32.tflite
```

### 9.2 可视化验证

使用 [Netron](https://netron.app/) 查看模型结构：

```
1. 打开 sample_sim.onnx → 查看原始结构
2. 打开 sample_sim_float32.tflite → 查看转换后结构
3. 对比节点类型、连接关系是否一致
```

---

## 十、常见问题排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 转换失败 | ONNX 算子不支持 | 检查算子兼容性 |
| 输出差异 | 数据格式未转换 | 检查 NCHW/NHWC 转换 |
| 模型变大 | 优化未生效 | 重新运行 onnx-simplifier |
| onnx2tf 下载失败 | 网络问题 | 创建本地缓存文件 |

---

## 参考文献与资源

### 官方文档与工具
- [onnx2tf GitHub](https://github.com/PINTO0309/onnx2tf) - ONNX 到 TFLite 转换工具
- [ONNX 官方文档](https://onnx.ai/) - Open Neural Network Exchange 官方网站
- [TensorFlow Lite 文档](https://www.tensorflow.org/lite) - TFLite 官方文档

### 技术博客与教程
- [ONNX 转 TFLite 详解 - CSDN](https://blog.csdn.net/gitblog_00303/article/details/157982232) - ONNX 模型转换为 TFLite 的详细教程
- [Silicon Labs MLTK 教程](https://siliconlabs.github.io/mltk/mltk/tutorials/onnx_to_tflite.html) - MLTK 中的 ONNX 到 TFLite 转换指南

### 硬件厂商文档
- [Qualcomm Hexagon NN 指南](https://docs.qualcomm.com/doc/80-80022-15B/topic/export-onnx-model-to-litert.html) - 导出 ONNX 模型到 LiteRT 的 Qualcomm 文档

### 可视化工具
- [Netron](https://netron.app/) - 模型结构可视化工具

### 其他资源
- [onnx-simplifier](https://github.com/daquexian/onnx-simplifier) - ONNX 模型简化工具
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - ONNX 推理引擎