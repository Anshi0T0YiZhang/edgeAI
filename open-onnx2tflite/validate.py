"""
模型验证脚本

验证 ONNX → TFLite 转换和 INT8 量化的精度，对比 ONNX、float32 TFLite、INT8 TFLite 三者输出。
参照 test-onnx2tflite/test_model.py 的推理与对比逻辑。

使用方法:
    python validate.py --all                                  # 验证全部模型
    python validate.py --model mnist-12                       # 验证单模型
    python validate.py --model mobilenetv2-7 --num-samples 5  # 指定验证样本数
"""

import argparse
import os
import struct
import sys
import time

import numpy as np


# ── 模型配置表（与 convert.py 一致）─────────────────────────

MODELS = {
    'mnist-12': {
        'onnx_file': 'mnist-12.onnx',
        'input_name': 'Input3',
        'input_shape_nchw': [1, 1, 28, 28],
        'calib_npy': 'data/calib_mnist_28x28x1_float32.npy',
        'dataset': 'mnist',
        'num_classes': 10,
    },
    'squeezenet1.1-7': {
        'onnx_file': 'squeezenet1.1-7.onnx',
        'input_name': 'data',
        'input_shape_nchw': [1, 3, 224, 224],
        'calib_npy': 'data/calib_imagenet_224x224x3_float32.npy',
        'dataset': 'imagenet',
        'num_classes': 1000,
    },
    'mobilenetv2-7': {
        'onnx_file': 'mobilenetv2-7.onnx',
        'input_name': 'data',
        'input_shape_nchw': [1, 3, 224, 224],
        'calib_npy': 'data/calib_imagenet_224x224x3_float32.npy',
        'dataset': 'imagenet',
        'num_classes': 1000,
    },
    'resnet18-v1-7': {
        'onnx_file': 'resnet18-v1-7.onnx',
        'input_name': 'data',
        'input_shape_nchw': [1, 3, 224, 224],
        'calib_npy': 'data/calib_imagenet_224x224x3_float32.npy',
        'dataset': 'imagenet',
        'num_classes': 1000,
    },
}

MODELS_DIR = 'models'


# ── ONNX 推理（参照 test-onnx2tflite/test_model.py）─────────

def test_onnx_inference(model_path: str, input_data: np.ndarray, input_name: str):
    """使用 ONNX Runtime 进行推理（逐样本推理以适配 batch=1 模型）"""
    import onnxruntime as ort

    session = ort.InferenceSession(model_path)

    all_outputs = []
    total_time = 0.0
    for i in range(input_data.shape[0]):
        single_input = input_data[i:i+1]  # [1, ...]
        start_time = time.time()
        outputs = session.run(None, {input_name: single_input})
        total_time += (time.time() - start_time) * 1000
        all_outputs.append(outputs[0])

    combined = np.concatenate(all_outputs, axis=0)

    return {
        'output': combined,
        'inference_time': total_time,
    }


# ── TFLite 推理（参照 test-onnx2tflite/test_model.py）────────

def quantize_input(float_data: np.ndarray, input_detail: dict) -> np.ndarray:
    """将 float32 数据量化为模型所需的 int8/uint8 格式"""
    dtype = input_detail['dtype']
    if dtype == np.float32:
        return float_data.astype(np.float32)

    scale = input_detail['quantization_parameters']['scales'][0]
    zp = input_detail['quantization_parameters']['zero_points'][0]
    # quantized = round(float / scale + zero_point)
    quantized = np.round(float_data / scale + zp).astype(dtype)
    return quantized


def dequantize_output(quant_data: np.ndarray, output_detail: dict) -> np.ndarray:
    """将 int8/uint8 输出反量化为 float32"""
    if quant_data.dtype == np.float32:
        return quant_data

    scale = output_detail['quantization_parameters']['scales'][0]
    zp = output_detail['quantization_parameters']['zero_points'][0]
    # float = (quantized - zero_point) * scale
    return (quant_data.astype(np.float32) - zp) * scale


def test_tflite_inference(model_path: str, input_data: np.ndarray):
    """使用 TFLite Interpreter 进行推理（逐样本推理，正确处理量化输入/输出）"""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    all_outputs = []
    total_time = 0.0
    for i in range(input_data.shape[0]):
        single_input = input_data[i:i+1]  # [1, ...]
        single_input = quantize_input(single_input, input_details[0])

        interpreter.set_tensor(input_details[0]['index'], single_input)

        start_time = time.time()
        interpreter.invoke()
        total_time += (time.time() - start_time) * 1000

        raw_output = interpreter.get_tensor(output_details[0]['index'])
        float_output = dequantize_output(raw_output, output_details[0])
        all_outputs.append(float_output)

    combined = np.concatenate(all_outputs, axis=0)

    return {
        'output': combined,
        'inference_time': total_time,
    }


# ── 对比输出（参照 test-onnx2tflite/test_model.py）───────────

def compare_outputs(onnx_output: np.ndarray, tflite_output: np.ndarray, label: str):
    """对比 ONNX 和 TFLite 的输出"""
    diff = np.abs(onnx_output - tflite_output)

    # Top-1 / Top-5 分类一致率
    onnx_top1 = np.argmax(onnx_output, axis=-1)
    tflite_top1 = np.argmax(tflite_output, axis=-1)
    top1_match = np.mean(onnx_top1 == tflite_top1) * 100

    onnx_top5 = np.argsort(onnx_output, axis=-1)[:, -5:]
    tflite_top5 = np.argsort(tflite_output, axis=-1)[:, -5:]
    top5_match = np.mean([set(a) == set(b) for a, b in zip(onnx_top5, tflite_top5)]) * 100

    print(f"  {label}:")
    print(f"    最大绝对误差: {diff.max():.6f}")
    print(f"    平均绝对误差: {diff.mean():.6f}")
    print(f"    Top-1 一致率: {top1_match:.1f}%")
    print(f"    Top-5 一致率: {top5_match:.1f}%")

    return {
        'max_abs_error': diff.max(),
        'mean_abs_error': diff.mean(),
        'top1_match': top1_match,
        'top5_match': top5_match,
    }


# ── 加载验证数据 ─────────────────────────────────────────────

def read_mnist_images(path: str) -> np.ndarray:
    """读取 MNIST idx 图像文件"""
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows, cols)


def load_validation_data(config: dict, num_samples: int) -> tuple:
    """加载验证数据，返回 (onnx_input_nchw, tflite_input_nhwc)"""
    dataset = config['dataset']

    if dataset == 'mnist':
        src = 'data/mnist/t10k-images-idx3-ubyte'
        images = read_mnist_images(src)[:num_samples]
        # 归一化到 0~1
        data = images.astype(np.float32) / 255.0
        # NCHW: [N, 1, 28, 28]
        onnx_input = data[:, np.newaxis, :, :]
        # NHWC: [N, 28, 28, 1]
        tflite_input = data[:, :, :, np.newaxis]

    elif dataset == 'imagenet':
        from PIL import Image
        import glob

        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        src_dir = 'data/imagenette2-320/val'
        files = sorted(glob.glob(os.path.join(src_dir, '*', '*.JPEG')))[:num_samples]

        def preprocess(path):
            img = Image.open(path).convert('RGB')
            w, h = img.size
            s = 256.0 / min(w, h)
            img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
            w, h = img.size
            left, top = (w - 224) // 2, (h - 224) // 2
            img = img.crop((left, top, left + 224, top + 224))
            x = np.asarray(img, dtype=np.float32) / 255.0
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            return x  # [224, 224, 3] HWC

        nhwc_data = np.stack([preprocess(f) for f in files], axis=0)  # [N, 224, 224, 3]
        # NCHW: [N, 3, 224, 224]
        onnx_input = np.transpose(nhwc_data, (0, 3, 1, 2))
        tflite_input = nhwc_data
    else:
        raise ValueError(f"未知数据集: {dataset}")

    return onnx_input, tflite_input


# ── 查找 TFLite 文件 ─────────────────────────────────────────

def find_tflite(output_dir: str, suffix: str = None) -> list:
    """查找输出目录中的 .tflite 文件"""
    results = []
    if not os.path.exists(output_dir):
        return results
    for root, _, files in os.walk(output_dir):
        for f in files:
            if f.endswith('.tflite'):
                if suffix is None or suffix in f:
                    results.append(os.path.join(root, f))
    return results


# ── 验证单个模型 ──────────────────────────────────────────────

def validate_model(name: str, config: dict, num_samples: int = 5):
    """验证单个模型的转换与量化精度"""
    onnx_path = os.path.join(MODELS_DIR, config['onnx_file'])

    if not os.path.exists(onnx_path):
        print(f"  [跳过] ONNX 模型不存在: {onnx_path}")
        return None

    # 加载验证数据
    print("  加载验证数据...")
    onnx_input, tflite_input = load_validation_data(config, num_samples)
    print(f"    ONNX 输入: {onnx_input.shape} ({onnx_input.dtype})")
    print(f"    TFLite 输入: {tflite_input.shape} ({tflite_input.dtype})")

    # ONNX 推理
    print("  ONNX Runtime 推理...")
    onnx_result = test_onnx_inference(onnx_path, onnx_input, config['input_name'])
    onnx_output = onnx_result['output']
    print(f"    输出形状: {onnx_output.shape}, 耗时: {onnx_result['inference_time']:.2f} ms")

    results = {'onnx_time': onnx_result['inference_time']}

    # 验证 float32 TFLite
    f32_dir = os.path.join(MODELS_DIR, f'{name}_float32')
    f32_files = find_tflite(f32_dir, 'float32')
    if f32_files:
        f32_path = f32_files[0]
        print(f"  float32 TFLite 推理 ({os.path.basename(f32_path)})...")
        f32_result = test_tflite_inference(f32_path, tflite_input)
        f32_output = f32_result['output']
        print(f"    输出形状: {f32_output.shape}, 耗时: {f32_result['inference_time']:.2f} ms")
        results['float32'] = compare_outputs(onnx_output, f32_output, 'float32 vs ONNX')
        results['float32_time'] = f32_result['inference_time']
    else:
        print("  [跳过] 未找到 float32 TFLite")

    # 验证 INT8 TFLite（full_integer_quant 为标准 INT8 量化）
    int8_dir = os.path.join(MODELS_DIR, f'{name}_int8')
    int8_files = find_tflite(int8_dir, 'full_integer_quant')
    # 排除 int16_act 版本
    int8_files = [f for f in int8_files if 'int16_act' not in f]
    if int8_files:
        int8_path = int8_files[0]
        print(f"  INT8 TFLite 推理 ({os.path.basename(int8_path)})...")
        int8_result = test_tflite_inference(int8_path, tflite_input)
        int8_output = int8_result['output']
        print(f"    输出形状: {int8_output.shape}, 耗时: {int8_result['inference_time']:.2f} ms")
        results['int8'] = compare_outputs(onnx_output, int8_output, 'INT8 vs ONNX')
        results['int8_time'] = int8_result['inference_time']
    else:
        print("  [跳过] 未找到 INT8 full_integer_quant TFLite")

    # 文件大小对比
    print("\n  文件大小对比:")
    onnx_size = os.path.getsize(onnx_path) / 1024
    def fmt(kb): return f"{kb/1024:.1f} MB" if kb > 1024 else f"{kb:.1f} KB"
    print(f"    ONNX:    {fmt(onnx_size)}")

    for qtype, suffix in [('float32', 'float32'), ('int8', 'full_integer_quant')]:
        qdir = os.path.join(MODELS_DIR, f'{name}_{qtype}')
        qfiles = find_tflite(qdir, suffix)
        qfiles = [f for f in qfiles if 'int16_act' not in f]
        for f in qfiles:
            size_kb = os.path.getsize(f) / 1024
            print(f"    {qtype}: {fmt(size_kb)}")

    return results


# ── 主函数 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='验证 ONNX/TFLite/INT8 模型精度')
    parser.add_argument('--model', type=str, choices=list(MODELS.keys()),
                        help='指定模型名称')
    parser.add_argument('--all', action='store_true',
                        help='验证全部模型')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='验证样本数 (默认 5)')
    args = parser.parse_args()

    # 选择目标模型
    if args.model:
        targets = {args.model: MODELS[args.model]}
    else:
        targets = MODELS

    print(f"验证样本数: {args.num_samples}")
    print(f"目标模型: {', '.join(targets.keys())}")
    print()

    all_results = {}

    for name, config in targets.items():
        print(f"{'='*60}")
        print(f"模型: {name}")
        print(f"{'='*60}")

        try:
            result = validate_model(name, config, args.num_samples)
            all_results[name] = result
        except Exception as e:
            print(f"  [失败] {name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = None
        print()

    # 汇总
    print(f"{'='*60}")
    print("验证汇总")
    print(f"{'='*60}")
    for name, result in all_results.items():
        if result is None:
            print(f"  {name}: 失败")
            continue
        print(f"  {name}:")
        if 'float32' in result:
            r = result['float32']
            print(f"    float32: max_err={r['max_abs_error']:.6f}, top1={r['top1_match']:.0f}%")
        if 'int8' in result:
            r = result['int8']
            print(f"    INT8:    max_err={r['max_abs_error']:.6f}, top1={r['top1_match']:.0f}%")

    return 0


if __name__ == '__main__':
    sys.exit(main())
