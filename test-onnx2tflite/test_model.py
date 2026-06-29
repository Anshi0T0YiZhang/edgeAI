"""
模型推理测试脚本

该脚本用于验证 ONNX 和 TFLite 模型的推理功能，
并对比两种格式的输出一致性。

使用方法:
    python test_model.py                    # 测试默认模型
    python test_model.py --onnx model.onnx  # 测试指定 ONNX 模型
    python test_model.py --tflite model.tflite  # 测试指定 TFLite 模型
    python test_model.py --compare          # 对比 ONNX 和 TFLite 输出
"""

import argparse
import os
import sys
import time
import numpy as np


def test_onnx_inference(model_path: str, input_data: np.ndarray = None):
    """使用 ONNX Runtime 进行推理"""
    print(f"\n{'='*50}")
    print(f"ONNX Runtime 推理测试")
    print(f"{'='*50}")

    try:
        import onnxruntime as ort
    except ImportError:
        print("错误: onnxruntime 未安装")
        print("请运行: pip install onnxruntime")
        return None

    # 加载模型
    print(f"加载模型: {model_path}")
    session = ort.InferenceSession(model_path)

    # 获取输入输出信息
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print(f"  输入名称: {input_info.name}")
    print(f"  输入形状: {input_info.shape}")
    print(f"  输出名称: {output_info.name}")
    print(f"  输出形状: {output_info.shape}")

    # 准备输入数据
    if input_data is None:
        input_shape = [dim if isinstance(dim, int) else 1 for dim in input_info.shape]
        input_data = np.random.randn(*input_shape).astype(np.float32)

    # 推理
    print(f"\n执行推理...")
    start_time = time.time()
    outputs = session.run(None, {input_info.name: input_data})
    inference_time = (time.time() - start_time) * 1000

    output = outputs[0]
    print(f"  输出形状: {output.shape}")
    print(f"  输出范围: [{output.min():.6f}, {output.max():.6f}]")
    print(f"  推理时间: {inference_time:.2f} ms")

    return {
        'output': output,
        'input_name': input_info.name,
        'output_name': output_info.name,
        'inference_time': inference_time
    }


def test_tflite_inference(model_path: str, input_data: np.ndarray = None):
    """使用 TFLite Interpreter 进行推理"""
    print(f"\n{'='*50}")
    print(f"TFLite Interpreter 推理测试")
    print(f"{'='*50}")

    try:
        import tensorflow as tf
    except ImportError:
        print("错误: tensorflow 未安装")
        print("请运行: pip install tensorflow")
        return None

    # 加载模型
    print(f"加载模型: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 获取输入输出信息
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_info = input_details[0]
    output_info = output_details[0]

    print(f"  输入名称: {input_info['name']}")
    print(f"  输入形状: {input_info['shape']}")
    print(f"  输出名称: {output_info['name']}")
    print(f"  输出形状: {output_info['shape']}")

    # 准备输入数据
    if input_data is None:
        input_shape = input_info['shape']
        input_data = np.random.randn(*input_shape).astype(np.float32)

    # 设置输入
    interpreter.set_tensor(input_info['index'], input_data)

    # 推理
    print(f"\n执行推理...")
    start_time = time.time()
    interpreter.invoke()
    inference_time = (time.time() - start_time) * 1000

    # 获取输出
    output = interpreter.get_tensor(output_info['index'])
    print(f"  输出形状: {output.shape}")
    print(f"  输出范围: [{output.min():.6f}, {output.max():.6f}]")
    print(f"  推理时间: {inference_time:.2f} ms")

    return {
        'output': output,
        'input_name': input_info['name'],
        'output_name': output_info['name'],
        'inference_time': inference_time
    }


def compare_outputs(onnx_result: dict, tflite_result: dict, tolerance: float = 1e-4):
    """对比 ONNX 和 TFLite 的输出"""
    print(f"\n{'='*50}")
    print(f"输出对比")
    print(f"{'='*50}")

    onnx_output = onnx_result['output']
    tflite_output = tflite_result['output']

    # 计算差异
    diff = np.abs(onnx_output - tflite_output)

    print(f"  最大绝对误差: {diff.max():.6f}")
    print(f"  平均绝对误差: {diff.mean():.6f}")
    print(f"  容差阈值: {tolerance}")

    # 检查是否在容差范围内
    if diff.max() < tolerance:
        print(f"\n  ✓ 输出一致 (误差在容差范围内)")
        return True
    else:
        print(f"\n  ✗ 输出不一致 (误差超出容差范围)")
        return False


def benchmark_inference(model_path: str, model_type: str, num_runs: int = 100):
    """基准测试推理性能"""
    print(f"\n{'='*50}")
    print(f"性能基准测试 ({model_type})")
    print(f"{'='*50}")

    if model_type == 'onnx':
        import onnxruntime as ort
        session = ort.InferenceSession(model_path)
        input_info = session.get_inputs()[0]
        input_shape = [dim if isinstance(dim, int) else 1 for dim in input_info.shape]
        input_data = np.random.randn(*input_shape).astype(np.float32)

        times = []
        for _ in range(num_runs):
            start = time.time()
            session.run(None, {input_info.name: input_data})
            times.append((time.time() - start) * 1000)

    elif model_type == 'tflite':
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        input_data = np.random.randn(*input_shape).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        times = []
        for _ in range(num_runs):
            start = time.time()
            interpreter.invoke()
            times.append((time.time() - start) * 1000)

    times = np.array(times)
    print(f"  运行次数: {num_runs}")
    print(f"  平均时间: {times.mean():.2f} ms")
    print(f"  最小时间: {times.min():.2f} ms")
    print(f"  最大时间: {times.max():.2f} ms")
    print(f"  标准差: {times.std():.2f} ms")

    return times


def inspect_onnx_model(model_path: str):
    """检查 ONNX 模型信息"""
    print(f"\n{'='*50}")
    print(f"ONNX 模型信息")
    print(f"{'='*50}")

    import onnx

    model = onnx.load(model_path)

    print(f"  IR 版本: {model.ir_version}")
    print(f"  生成器: {model.producer_name} {model.producer_version}")

    # Opset 信息
    print(f"\n  Opset 导入:")
    for opset in model.opset_import:
        domain = opset.domain if opset.domain else "ai.onnx"
        print(f"    - {domain}: {opset.version}")

    # 输入信息
    print(f"\n  输入:")
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"    - {inp.name}: {shape}")

    # 输出信息
    print(f"\n  输出:")
    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"    - {out.name}: {shape}")

    # 节点信息
    print(f"\n  节点数量: {len(model.graph.node)}")
    op_types = {}
    for node in model.graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

    print(f"\n  算子统计:")
    for op_type, count in sorted(op_types.items()):
        print(f"    - {op_type}: {count}")


def inspect_tflite_model(model_path: str):
    """检查 TFLite 模型信息"""
    print(f"\n{'='*50}")
    print(f"TFLite 模型信息")
    print(f"{'='*50}")

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=model_path)

    # 输入信息
    print(f"\n  输入:")
    for inp in interpreter.get_input_details():
        print(f"    - {inp['name']}: {inp['shape']} ({inp['dtype']})")

    # 输出信息
    print(f"\n  输出:")
    for out in interpreter.get_output_details():
        print(f"    - {out['name']}: {out['shape']} ({out['dtype']})")

    # 张量信息
    print(f"\n  张量数量: {len(interpreter.get_tensor_details())}")

    # 算子信息
    print(f"\n  算子数量: {len(interpreter._get_ops_details())}")


def main():
    parser = argparse.ArgumentParser(description='ONNX/TFLite 模型测试工具')
    parser.add_argument('--onnx', type=str, default='models/onnx_model.onnx',
                       help='ONNX 模型路径')
    parser.add_argument('--tflite', type=str, default='models/keras_model.tflite',
                       help='TFLite 模型路径')
    parser.add_argument('--compare', '-c', action='store_true',
                       help='对比 ONNX 和 TFLite 输出')
    parser.add_argument('--benchmark', '-b', action='store_true',
                       help='运行性能基准测试')
    parser.add_argument('--inspect', '-i', action='store_true',
                       help='检查模型详细信息')
    parser.add_argument('--runs', type=int, default=100,
                       help='基准测试运行次数')

    args = parser.parse_args()

    # 自动查找 TFLite 模型
    if args.tflite is None:
        models_dir = os.path.dirname(args.onnx)
        for f in os.listdir(models_dir) if os.path.exists(models_dir) else []:
            if f.endswith('.tflite'):
                args.tflite = os.path.join(models_dir, f)
                break

    # 检查模型是否存在
    if not os.path.exists(args.onnx) and not args.tflite:
        print("错误: 未找到任何模型文件")
        print("请先运行 convert.py 创建模型")
        return 1

    # 检查模型信息
    if args.inspect:
        if os.path.exists(args.onnx):
            inspect_onnx_model(args.onnx)
        if args.tflite and os.path.exists(args.tflite):
            inspect_tflite_model(args.tflite)

    # 推理测试
    if args.compare and os.path.exists(args.onnx) and args.tflite and os.path.exists(args.tflite):
        # 生成相同的输入数据
        np.random.seed(42)
        # ONNX 使用 NCHW 格式 [N, C, H, W]
        input_data_nchw = np.random.randn(1, 3, 64, 64).astype(np.float32)
        # TFLite 使用 NHWC 格式 [N, H, W, C]，需要转换
        input_data_nhwc = np.transpose(input_data_nchw, (0, 2, 3, 1))

        onnx_result = test_onnx_inference(args.onnx, input_data_nchw)
        tflite_result = test_tflite_inference(args.tflite, input_data_nhwc)

        if onnx_result and tflite_result:
            compare_outputs(onnx_result, tflite_result)
    else:
        # 单独测试
        if os.path.exists(args.onnx):
            test_onnx_inference(args.onnx)
        if args.tflite and os.path.exists(args.tflite):
            test_tflite_inference(args.tflite)

    # 性能测试
    if args.benchmark:
        if os.path.exists(args.onnx):
            benchmark_inference(args.onnx, 'onnx', args.runs)
        if args.tflite and os.path.exists(args.tflite):
            benchmark_inference(args.tflite, 'tflite', args.runs)

    print(f"\n{'='*50}")
    print("测试完成!")
    print(f"{'='*50}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
