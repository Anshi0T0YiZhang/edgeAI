"""
ONNX 模型下载脚本

从 HuggingFace onnxmodelzoo 下载公开 ONNX 模型。

使用方法:
    python download_models.py              # 下载全部模型
    python download_models.py --model mnist-12  # 下载指定模型
    python download_models.py --mirror     # 使用国内镜像
"""

import argparse
import os
import subprocess
import sys


# 模型配置表
MODELS = {
    'mnist-12': {
        'repo': 'onnxmodelzoo/mnist-12',
        'file': 'mnist-12.onnx',
    },
    'squeezenet1.1-7': {
        'repo': 'onnxmodelzoo/squeezenet1.1-7',
        'file': 'squeezenet1.1-7.onnx',
    },
    'mobilenetv2-7': {
        'repo': 'onnxmodelzoo/mobilenetv2-7',
        'file': 'mobilenetv2-7.onnx',
    },
    'resnet18-v1-7': {
        'repo': 'onnxmodelzoo/resnet18-v1-7',
        'file': 'resnet18-v1-7.onnx',
    },
}

MODELS_DIR = 'models'


def download_model(name: str, config: dict, use_mirror: bool = False):
    """下载单个 ONNX 模型"""
    dest = os.path.join(MODELS_DIR, config['file'])

    if os.path.exists(dest):
        size_kb = os.path.getsize(dest) / 1024
        print(f"  [跳过] {name}: {dest} 已存在 ({size_kb:.1f} KB)")
        return True

    if use_mirror:
        base_url = 'https://hf-mirror.com'
    else:
        base_url = 'https://huggingface.co'

    url = f"{base_url}/{config['repo']}/resolve/main/{config['file']}"

    print(f"  下载 {name} -> {dest}")
    print(f"    URL: {url}")

    result = subprocess.run(
        ['wget', '-q', '--show-progress', url, '-O', dest],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"  [失败] {name}: {result.stderr.strip()}")
        # 清理不完整文件
        if os.path.exists(dest):
            os.remove(dest)
        return False

    size_kb = os.path.getsize(dest) / 1024
    if size_kb > 1024:
        print(f"  [完成] {name}: {size_kb/1024:.1f} MB")
    else:
        print(f"  [完成] {name}: {size_kb:.1f} KB")
    return True


def main():
    parser = argparse.ArgumentParser(description='下载 ONNX 模型')
    parser.add_argument('--model', type=str, choices=list(MODELS.keys()),
                        help='指定模型名称')
    parser.add_argument('--all', action='store_true',
                        help='下载全部模型')
    parser.add_argument('--mirror', action='store_true',
                        help='使用 hf-mirror.com 国内镜像')
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.model:
        targets = {args.model: MODELS[args.model]}
    else:
        # 默认下载全部
        targets = MODELS

    print(f"将下载 {len(targets)} 个模型到 {MODELS_DIR}/")
    if args.mirror:
        print("使用国内镜像: hf-mirror.com")
    print()

    success = 0
    failed = []
    for name, config in targets.items():
        if download_model(name, config, args.mirror):
            success += 1
        else:
            failed.append(name)

    print(f"\n下载完成: {success}/{len(targets)} 成功")
    if failed:
        print(f"失败: {', '.join(failed)}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
