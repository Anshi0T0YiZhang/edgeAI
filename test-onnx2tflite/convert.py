"""
ONNX 到 TFLite 转换脚本

使用方法:
    python convert.py                    # 创建示例模型并转换
    python convert.py -i your_model.onnx # 转换指定模型
"""

import argparse
import os
import numpy as np


def create_sample_onnx(output_path: str):
    """创建一个简单的 ONNX 示例模型"""
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    print("创建示例 ONNX 模型...")

    conv1_w = numpy_helper.from_array(np.random.randn(16, 3, 3, 3).astype(np.float32) * 0.1, 'conv1_w')
    conv1_b = numpy_helper.from_array(np.zeros(16, dtype=np.float32), 'conv1_b')
    conv2_w = numpy_helper.from_array(np.random.randn(32, 16, 3, 3).astype(np.float32) * 0.1, 'conv2_w')
    conv2_b = numpy_helper.from_array(np.zeros(32, dtype=np.float32), 'conv2_b')
    fc_w = numpy_helper.from_array(np.random.randn(32, 10).astype(np.float32) * 0.1, 'fc_w')
    fc_b = numpy_helper.from_array(np.zeros(10, dtype=np.float32), 'fc_b')
    reshape_shape = numpy_helper.from_array(np.array([1, 32], dtype=np.int64), 'reshape_shape')

    nodes = [
        helper.make_node('Conv', ['input', 'conv1_w', 'conv1_b'], ['c1'], kernel_shape=[3,3], pads=[1,1,1,1]),
        helper.make_node('Relu', ['c1'], ['r1']),
        helper.make_node('MaxPool', ['r1'], ['p1'], kernel_shape=[2,2], strides=[2,2]),
        helper.make_node('Conv', ['p1', 'conv2_w', 'conv2_b'], ['c2'], kernel_shape=[3,3], pads=[1,1,1,1]),
        helper.make_node('Relu', ['c2'], ['r2']),
        helper.make_node('GlobalAveragePool', ['r2'], ['gap']),
        helper.make_node('Reshape', ['gap', 'reshape_shape'], ['flat']),
        helper.make_node('MatMul', ['flat', 'fc_w'], ['fc']),
        helper.make_node('Add', ['fc', 'fc_b'], ['output']),
    ]

    graph = helper.make_graph(nodes, 'simple_cnn',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10])],
        [conv1_w, conv1_b, conv2_w, conv2_b, fc_w, fc_b, reshape_shape])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"ONNX: {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")


def simplify_onnx(input_path: str, output_path: str):
    """简化 ONNX 模型"""
    import onnxsim, onnx
    print("简化 ONNX...")
    model = onnx.load(input_path)
    model_simp, _ = onnxsim.simplify(model)
    onnx.save(model_simp, output_path)
    print(f"简化: {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")


def onnx_to_tflite(onnx_path: str, output_dir: str):
    """使用 onnx2tf 将 ONNX 转换为 TFLite"""
    import onnx2tf

    # 创建本地缓存文件（绕过 onnx2tf 的下载 bug）
    cache_file = 'calibration_image_sample_data_20x128x128x3_float32.npy'
    if not os.path.exists(cache_file):
        np.save(cache_file, np.random.rand(20, 128, 128, 3).astype(np.float32))

    print("转换 ONNX -> TFLite...")
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=output_dir,
        copy_onnx_input_output_names_to_tflite=True,
    )

    # 查找生成的 tflite 文件
    tflite_files = [f for f in os.listdir(output_dir) if f.endswith('.tflite')]
    for f in tflite_files:
        print(f"TFLite: {os.path.join(output_dir, f)} ({os.path.getsize(os.path.join(output_dir, f))/1024:.1f} KB)")
    return tflite_files


def main():
    parser = argparse.ArgumentParser(description='ONNX 到 TFLite 转换')
    parser.add_argument('-i', '--input', help='输入 ONNX 模型')
    parser.add_argument('-o', '--output', default='models', help='输出目录')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.input:
        onnx_path = args.input
    else:
        onnx_path = os.path.join(args.output, 'sample.onnx')
        create_sample_onnx(onnx_path)

    simplified_path = onnx_path.replace('.onnx', '_sim.onnx')
    simplify_onnx(onnx_path, simplified_path)

    tflite_files = onnx_to_tflite(simplified_path, args.output)

    if tflite_files:
        print(f"\n转换完成!")
        print(f"  ONNX:  {onnx_path}")
        for f in tflite_files:
            print(f"  TFLite: {os.path.join(args.output, f)}")


if __name__ == '__main__':
    main()