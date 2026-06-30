"""
ONNX 到 TFLite 转换脚本（支持 float32 和 INT8 量化）

参照 test-onnx2tflite/convert.py 的转换逻辑，扩展支持多模型和 INT8 量化。
由于 onnx2tf Python API 存在 segfault 问题，改用 subprocess 调用 CLI。

使用方法:
    python convert.py --all                                  # 全部模型 float32 转换
    python convert.py --model mnist-12                       # 单模型 float32 转换
    python convert.py --all --quantize int8                  # 全部模型 INT8 量化
    python convert.py --model mobilenetv2-7 --quantize int8  # 单模型 INT8 量化
"""

import argparse
import os
import subprocess
import sys


# ── 模型配置表 ──────────────────────────────────────────────

MODELS = {
    'mnist-12': {
        'onnx_file': 'mnist-12.onnx',
        'input_name': 'Input3',
        'input_shape': [1, 1, 28, 28],
        'calib_npy': 'data/calib_mnist_28x28x1_float32.npy',
        'cind_mean': '[[[[0.1307]]]]',
        'cind_std': '[[[[0.3081]]]]',
    },
    'squeezenet1.1-7': {
        'onnx_file': 'squeezenet1.1-7.onnx',
        'input_name': 'data',
        'input_shape': [1, 3, 224, 224],
        'calib_npy': 'data/calib_imagenet_224x224x3_float32.npy',
        'cind_mean': '[[[[0.485,0.456,0.406]]]]',
        'cind_std': '[[[[0.229,0.224,0.225]]]]',
    },
    'mobilenetv2-7': {
        'onnx_file': 'mobilenetv2-7.onnx',
        'input_name': 'data',
        'input_shape': [1, 3, 224, 224],
        'calib_npy': 'data/calib_imagenet_224x224x3_float32.npy',
        'cind_mean': '[[[[0.485,0.456,0.406]]]]',
        'cind_std': '[[[[0.229,0.224,0.225]]]]',
    },
    'resnet18-v1-7': {
        'onnx_file': 'resnet18-v1-7.onnx',
        'input_name': 'data',
        'input_shape': [1, 3, 224, 224],
        'calib_npy': 'data/calib_imagenet_224x224x3_float32.npy',
        'cind_mean': '[[[[0.485,0.456,0.406]]]]',
        'cind_std': '[[[[0.229,0.224,0.225]]]]',
    },
}

MODELS_DIR = 'models'


# ── 辅助函数 ────────────────────────────────────────────────

def simplify_onnx(input_path: str, output_path: str):
    """简化 ONNX 模型（参照 test-onnx2tflite/convert.py）"""
    import onnxsim
    import onnx

    print("    简化 ONNX...")
    model = onnx.load(input_path)
    model_simp, _ = onnxsim.simplify(model)
    onnx.save(model_simp, output_path)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"    简化完成: {output_path} ({size_kb:.1f} KB)")


def find_tflite(output_dir: str) -> list:
    """查找输出目录中的所有 .tflite 文件"""
    results = []
    for root, _, files in os.walk(output_dir):
        for f in files:
            if f.endswith('.tflite'):
                results.append(os.path.join(root, f))
    return results


def convert_float32(name: str, config: dict):
    """ONNX → float32 TFLite（使用 onnx2tf CLI）"""
    onnx_path = os.path.join(MODELS_DIR, config['onnx_file'])
    simplified_path = onnx_path.replace('.onnx', '_sim.onnx')
    output_dir = os.path.join(MODELS_DIR, f'{name}_float32')

    # 检查是否已有输出
    existing = find_tflite(output_dir)
    if existing:
        for f in existing:
            size_kb = os.path.getsize(f) / 1024
            print(f"  [跳过] {name} TFLite 已存在: {os.path.basename(f)} ({size_kb:.1f} KB)")
        return True

    # 简化
    if not os.path.exists(simplified_path):
        simplify_onnx(onnx_path, simplified_path)

    # 使用 onnx2tf CLI 转换
    print("    转换 ONNX -> float32 TFLite (CLI)...")
    cmd = [
        'onnx2tf',
        '-i', simplified_path,
        '-o', output_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"    [失败] onnx2tf 返回码: {result.returncode}")
        print(f"    stderr: {result.stderr[-500:]}" if result.stderr else "")
        return False

    # 查找并报告输出
    tflite_files = find_tflite(output_dir)
    if not tflite_files:
        print(f"    [失败] {name}: 未生成 TFLite 文件")
        return False

    for f in tflite_files:
        size_kb = os.path.getsize(f) / 1024
        if size_kb > 1024:
            print(f"    [完成] {os.path.basename(f)}: {size_kb/1024:.1f} MB")
        else:
            print(f"    [完成] {os.path.basename(f)}: {size_kb:.1f} KB")
    return True


def convert_int8(name: str, config: dict):
    """ONNX → INT8 TFLite（使用 onnx2tf CLI + -oiqt + -cind）"""
    onnx_path = os.path.join(MODELS_DIR, config['onnx_file'])
    simplified_path = onnx_path.replace('.onnx', '_sim.onnx')
    output_dir = os.path.join(MODELS_DIR, f'{name}_int8')
    calib_npy = config['calib_npy']

    # 检查校准文件
    if not os.path.exists(calib_npy):
        print(f"  [失败] {name}: 校准文件不存在: {calib_npy}")
        print("  请先运行: python build_calib.py")
        return False

    # 检查是否已有输出
    existing = find_tflite(output_dir)
    if existing:
        for f in existing:
            size_kb = os.path.getsize(f) / 1024
            print(f"  [跳过] {name} INT8 TFLite 已存在: {os.path.basename(f)} ({size_kb:.1f} KB)")
        return True

    # 简化（若尚未简化）
    if not os.path.exists(simplified_path):
        simplify_onnx(onnx_path, simplified_path)

    # INT8 量化转换
    print("    转换 ONNX -> INT8 TFLite (CLI)...")
    cmd = [
        'onnx2tf',
        '-i', simplified_path,
        '-o', output_dir,
        '-oiqt',
        '-cind', config['input_name'], calib_npy, config['cind_mean'], config['cind_std'],
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"    [失败] onnx2tf 返回码: {result.returncode}")
        print(f"    stderr: {result.stderr[-500:]}" if result.stderr else "")
        return False

    # 查找并报告输出
    tflite_files = find_tflite(output_dir)
    if not tflite_files:
        print(f"    [失败] {name}: 未生成 INT8 TFLite 文件")
        return False

    for f in tflite_files:
        size_kb = os.path.getsize(f) / 1024
        if size_kb > 1024:
            print(f"    [完成] {os.path.basename(f)}: {size_kb/1024:.1f} MB")
        else:
            print(f"    [完成] {os.path.basename(f)}: {size_kb:.1f} KB")
    return True


# ── 主函数 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ONNX 到 TFLite 转换（支持 INT8 量化）')
    parser.add_argument('--model', type=str, choices=list(MODELS.keys()),
                        help='指定模型名称')
    parser.add_argument('--all', action='store_true',
                        help='转换全部模型')
    parser.add_argument('--quantize', type=str, choices=['float32', 'int8'],
                        default='float32',
                        help='量化方式: float32 (默认) 或 int8')
    args = parser.parse_args()

    # 选择目标模型
    if args.model:
        targets = {args.model: MODELS[args.model]}
    else:
        targets = MODELS

    quant_label = 'INT8' if args.quantize == 'int8' else 'float32'
    print(f"转换模式: {quant_label}")
    print(f"目标模型: {', '.join(targets.keys())}")
    print()

    success = 0
    failed = []

    for name, config in targets.items():
        print(f"{'='*50}")
        print(f"模型: {name} ({quant_label})")
        print(f"{'='*50}")

        try:
            if args.quantize == 'int8':
                ok = convert_int8(name, config)
            else:
                ok = convert_float32(name, config)
        except Exception as e:
            print(f"  [失败] {name}: {e}")
            ok = False

        if ok:
            success += 1
        else:
            failed.append(name)
        print()

    print(f"{'='*50}")
    print(f"转换完成: {success}/{len(targets)} 成功")
    if failed:
        print(f"失败: {', '.join(failed)}")
        return 1

    # 打印结果汇总
    print(f"\n{'='*50}")
    print("文件大小汇总:")
    print(f"{'='*50}")
    for name in targets:
        onnx_path = os.path.join(MODELS_DIR, MODELS[name]['onnx_file'])

        onnx_size = os.path.getsize(onnx_path) / 1024 if os.path.exists(onnx_path) else 0

        def fmt(kb):
            return f"{kb/1024:.1f} MB" if kb > 1024 else f"{kb:.1f} KB"

        print(f"  {name}:")
        print(f"    ONNX:      {fmt(onnx_size)}")

        for qtype in ['float32', 'int8']:
            out_dir = os.path.join(MODELS_DIR, f'{name}_{qtype}')
            tflite_files = find_tflite(out_dir) if os.path.exists(out_dir) else []
            for tf_file in tflite_files:
                size_kb = os.path.getsize(tf_file) / 1024
                print(f"    {qtype}:     {os.path.basename(tf_file)} -> {fmt(size_kb)}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
