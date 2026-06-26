"""
ONNX 与 TFLite 模型转换演示脚本

由于 onnx2tf 库与 numpy 存在兼容性问题，本脚本采用替代方案：
使用 TensorFlow/Keras 创建模型，同时导出为 ONNX 和 TFLite 格式

这样可以演示两种格式的创建和使用过程

使用方法:
    python convert.py                    # 创建并导出模型
    python convert.py --quantize float16 # 导出并量化
"""

import argparse
import os
import sys
import numpy as np


def create_keras_model():
    """使用 Keras 创建一个简单的 CNN 模型"""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    print("正在创建 Keras CNN 模型...")

    # 创建一个简单的 CNN 模型
    model = keras.Sequential([
        layers.Input(shape=(64, 64, 3)),  # NHWC 格式 (TensorFlow 默认)
        layers.Conv2D(16, 3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(10)
    ])

    model.summary()

    return model


def export_to_tflite(model, output_path: str, quantize: str = None):
    """将 Keras 模型导出为 TFLite 格式"""
    import tensorflow as tf

    print(f"\n正在导出 TFLite 模型...")

    # 创建转换器
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 量化选项
    if quantize == 'float16':
        print("  应用 Float16 量化...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == 'int8':
        print("  应用 INT8 量化...")
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(1, 64, 64, 3).astype(np.float32)
                yield [data]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    # 转换
    tflite_model = converter.convert()

    # 保存
    if quantize:
        output_path = output_path.replace('.tflite', f'_quant_{quantize}.tflite')

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"  TFLite 模型已保存到: {output_path}")
    print(f"  模型大小: {os.path.getsize(output_path) / 1024:.2f} KB")

    return output_path


def export_to_onnx(model, output_path: str):
    """将 Keras 模型导出为 ONNX 格式"""
    import tf2onnx
    import tensorflow as tf

    print(f"\n正在导出 ONNX 模型...")

    # 使用 tf2onnx 转换
    # 首先保存为 SavedModel
    saved_model_path = output_path.replace('.onnx', '_saved_model')
    model.save(saved_model_path)

    # 转换为 ONNX
    onnx_model, _ = tf2onnx.convert.from_saved_model(
        saved_model_path,
        opset=11,
        output_path=output_path
    )

    print(f"  ONNX 模型已保存到: {output_path}")
    print(f"  模型大小: {os.path.getsize(output_path) / 1024:.2f} KB")

    # 清理临时 SavedModel
    import shutil
    shutil.rmtree(saved_model_path, ignore_errors=True)

    return output_path


def create_onnx_from_scratch(output_path: str):
    """直接使用 ONNX API 创建模型（不依赖 TensorFlow）"""
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    print("\n正在使用 ONNX API 创建模型...")

    # 模型参数 (NCHW 格式，ONNX 标准)
    input_channels = 3
    input_height = 64
    input_width = 64
    num_classes = 10

    # 卷积层权重
    conv1_weight = numpy_helper.from_array(
        np.random.randn(16, 3, 3, 3).astype(np.float32) * 0.1,
        name='conv1_weight'
    )
    conv1_bias = numpy_helper.from_array(
        np.zeros(16, dtype=np.float32),
        name='conv1_bias'
    )

    conv2_weight = numpy_helper.from_array(
        np.random.randn(32, 16, 3, 3).astype(np.float32) * 0.1,
        name='conv2_weight'
    )
    conv2_bias = numpy_helper.from_array(
        np.zeros(32, dtype=np.float32),
        name='conv2_bias'
    )

    # 全连接层权重
    fc_weight = numpy_helper.from_array(
        np.random.randn(32, 10).astype(np.float32) * 0.1,
        name='fc_weight'
    )
    fc_bias = numpy_helper.from_array(
        np.zeros(10, dtype=np.float32),
        name='fc_bias'
    )

    # Reshape 参数 - 将 [1, 32, 1, 1] 变为 [1, 32]
    reshape_shape = numpy_helper.from_array(
        np.array([1, 32], dtype=np.int64),
        name='reshape_shape'
    )

    # 构建计算图
    nodes = [
        helper.make_node('Conv', ['input', 'conv1_weight', 'conv1_bias'],
                        ['conv1_out'], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node('Relu', ['conv1_out'], ['relu1_out']),
        helper.make_node('MaxPool', ['relu1_out'], ['pool1_out'],
                        kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node('Conv', ['pool1_out', 'conv2_weight', 'conv2_bias'],
                        ['conv2_out'], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node('Relu', ['conv2_out'], ['relu2_out']),
        helper.make_node('GlobalAveragePool', ['relu2_out'], ['gap_out']),
        helper.make_node('Reshape', ['gap_out', 'reshape_shape'], ['reshape_out']),
        helper.make_node('MatMul', ['reshape_out', 'fc_weight'], ['fc_out']),
        helper.make_node('Add', ['fc_out', 'fc_bias'], ['output']),
    ]

    # 输入输出定义
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [1, input_channels, input_height, input_width]
    )
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [1, num_classes]
    )

    # 初始化器
    initializers = [conv1_weight, conv1_bias, conv2_weight, conv2_bias, fc_weight, fc_bias, reshape_shape]

    # 创建计算图和模型
    graph = helper.make_graph(nodes, 'simple_cnn', [input_tensor], [output_tensor], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 7

    # 验证
    onnx.checker.check_model(model)

    # 保存
    onnx.save(model, output_path)

    print(f"  ONNX 模型已保存到: {output_path}")
    print(f"  模型大小: {os.path.getsize(output_path) / 1024:.2f} KB")

    return output_path


def verify_onnx_model(model_path: str):
    """验证 ONNX 模型"""
    import onnx
    import onnxruntime as ort

    print(f"\n验证 ONNX 模型: {model_path}")

    # 加载模型
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    # 打印模型信息
    print(f"  IR 版本: {model.ir_version}")
    print(f"  Opset: {model.opset_import[0].version}")
    print(f"  输入: {[i.name for i in model.graph.input]}")
    print(f"  输出: {[o.name for o in model.graph.output]}")

    # 使用 ONNX Runtime 推理
    session = ort.InferenceSession(model_path)
    input_info = session.get_inputs()[0]

    # 创建测试输入
    input_shape = [dim if isinstance(dim, int) else 1 for dim in input_info.shape]
    test_input = np.random.randn(*input_shape).astype(np.float32)

    # 推理
    output = session.run(None, {input_info.name: test_input})[0]

    print(f"  测试推理成功!")
    print(f"  输入形状: {test_input.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")

    return True


def verify_tflite_model(model_path: str):
    """验证 TFLite 模型"""
    import tensorflow as tf

    print(f"\n验证 TFLite 模型: {model_path}")

    # 加载模型
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 获取输入输出信息
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  输入: {input_details[0]['name']}, 形状: {input_details[0]['shape']}")
    print(f"  输出: {output_details[0]['name']}, 形状: {output_details[0]['shape']}")

    # 创建测试输入
    input_shape = input_details[0]['shape']
    test_input = np.random.randn(*input_shape).astype(np.float32)

    # 推理
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    print(f"  测试推理成功!")
    print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")

    return True


def get_file_size(path: str) -> str:
    """获取文件大小"""
    size = os.path.getsize(path)
    return f"{size / 1024:.2f} KB"


def main():
    parser = argparse.ArgumentParser(description='ONNX/TFLite 模型创建与转换演示')
    parser.add_argument('--output', '-o', type=str, default='models',
                       help='输出目录')
    parser.add_argument('--quantize', '-q', type=str, choices=['int8', 'float16'],
                       default=None, help='TFLite 量化类型')
    parser.add_argument('--method', '-m', type=str, choices=['keras', 'onnx', 'both'],
                       default='both', help='创建方法')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("ONNX 与 TFLite 模型创建与转换演示")
    print("=" * 60)

    if args.method in ['keras', 'both']:
        # 方法 1: 使用 Keras 创建，同时导出 ONNX 和 TFLite
        print("\n【方法 1: 使用 TensorFlow/Keras】")
        print("-" * 40)

        model = create_keras_model()

        # 导出 TFLite
        tflite_path = os.path.join(args.output, 'keras_model.tflite')
        export_to_tflite(model, tflite_path, args.quantize)

        # 尝试导出 ONNX
        try:
            onnx_path = os.path.join(args.output, 'keras_model.onnx')
            export_to_onnx(model, onnx_path)
        except Exception as e:
            print(f"  ONNX 导出失败: {e}")
            print("  (这可能是 tf2onnx 版本兼容性问题)")

    if args.method in ['onnx', 'both']:
        # 方法 2: 直接使用 ONNX API 创建
        print("\n【方法 2: 使用 ONNX API】")
        print("-" * 40)

        onnx_path = os.path.join(args.output, 'onnx_model.onnx')
        create_onnx_from_scratch(onnx_path)

    # 验证模型
    print("\n" + "=" * 60)
    print("模型验证")
    print("=" * 60)

    # 验证 TFLite
    tflite_files = [f for f in os.listdir(args.output) if f.endswith('.tflite')]
    for f in tflite_files:
        verify_tflite_model(os.path.join(args.output, f))

    # 验证 ONNX
    onnx_files = [f for f in os.listdir(args.output) if f.endswith('.onnx')]
    for f in onnx_files:
        verify_onnx_model(os.path.join(args.output, f))

    # 总结
    print("\n" + "=" * 60)
    print("转换完成! 生成的文件:")
    print("=" * 60)

    all_files = os.listdir(args.output)
    for f in sorted(all_files):
        if not f.endswith('.npy'):  # 排除测试输入文件
            path = os.path.join(args.output, f)
            if os.path.isfile(path):
                print(f"  {f}: {get_file_size(path)}")

    return 0


if __name__ == '__main__':
    sys.exit(main())