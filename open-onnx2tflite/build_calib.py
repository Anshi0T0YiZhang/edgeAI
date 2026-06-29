"""
校准数据生成脚本

从真实数据集中提取样本，生成 NHWC float32 的 .npy 校准文件，
用于 onnx2tf 的 INT8 量化校准。

使用方法:
    python build_calib.py                    # 生成全部校准数据
    python build_calib.py --dataset mnist    # 仅生成 MNIST 校准数据
    python build_calib.py --dataset imagenet # 仅生成 ImageNet 校准数据
    python build_calib.py --num-samples 100  # 指定校准样本数
"""

import argparse
import os
import struct
import sys

import numpy as np


DATA_DIR = 'data'

# MNIST 预处理参数：仅归一化到 0~1
MNIST_MEAN = 0.0
MNIST_STD = 1.0

# ImageNet 预处理参数
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_mnist_images(path: str) -> np.ndarray:
    """读取 MNIST idx 图像文件，返回 [N, 28, 28] uint8"""
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows, cols)


def build_mnist_calib(num_samples: int = 200):
    """从 MNIST test 集生成校准 .npy"""
    src = os.path.join(DATA_DIR, 'mnist', 't10k-images-idx3-ubyte')
    out = os.path.join(DATA_DIR, 'calib_mnist_28x28x1_float32.npy')

    if os.path.exists(out):
        data = np.load(out)
        print(f"  [跳过] {out} 已存在，形状: {data.shape}")
        return True

    if not os.path.exists(src):
        print(f"  [失败] 找不到 MNIST 数据: {src}")
        print("  请先运行: python download_data.py --dataset mnist")
        return False

    print("  读取 MNIST test 集...")
    images = read_mnist_images(src)  # [N, 28, 28] uint8

    # 取前 num_samples 张
    images = images[:num_samples]

    # 预处理：归一化到 0~1，添加通道维 → [N, 28, 28, 1] NHWC float32
    data = images.astype(np.float32) / 255.0
    data = data[:, :, :, np.newaxis]  # [N, 28, 28, 1]

    np.save(out, data)
    print(f"  [保存] {out}: 形状 {data.shape}, dtype {data.dtype}")
    print(f"    值范围: [{data.min():.4f}, {data.max():.4f}]")
    return True


def build_imagenet_calib(num_samples: int = 200):
    """从 Imagenette val 集生成校准 .npy"""
    from PIL import Image

    src_dir = os.path.join(DATA_DIR, 'imagenette2-320', 'val')
    out = os.path.join(DATA_DIR, 'calib_imagenet_224x224x3_float32.npy')

    if os.path.exists(out):
        data = np.load(out)
        print(f"  [跳过] {out} 已存在，形状: {data.shape}")
        return True

    if not os.path.exists(src_dir):
        print(f"  [失败] 找不到 Imagenette 数据: {src_dir}")
        print("  请先运行: python download_data.py --dataset imagenette")
        return False

    # 收集所有 JPEG 文件
    import glob
    files = sorted(glob.glob(os.path.join(src_dir, '*', '*.JPEG')))
    if not files:
        print(f"  [失败] 在 {src_dir} 中未找到 JPEG 文件")
        return False

    files = files[:num_samples]
    print(f"  预处理 {len(files)} 张 Imagenette 图片...")

    def preprocess(path: str) -> np.ndarray:
        """ImageNet 标准预处理 → [224, 224, 3] float32"""
        img = Image.open(path).convert('RGB')
        # resize 短边到 256，再 center crop 224x224
        w, h = img.size
        s = 256.0 / min(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
        w, h = img.size
        left, top = (w - 224) // 2, (h - 224) // 2
        img = img.crop((left, top, left + 224, top + 224))
        # 归一化 + 标准化
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD  # [224, 224, 3]
        return x

    data = np.stack([preprocess(f) for f in files], axis=0)  # [N, 224, 224, 3]
    np.save(out, data)
    print(f"  [保存] {out}: 形状 {data.shape}, dtype {data.dtype}")
    print(f"    值范围: [{data.min():.4f}, {data.max():.4f}]")
    return True


def main():
    parser = argparse.ArgumentParser(description='生成 INT8 量化校准数据')
    parser.add_argument('--dataset', type=str,
                        choices=['mnist', 'imagenet'],
                        help='指定数据集')
    parser.add_argument('--num-samples', type=int, default=200,
                        help='校准样本数量 (默认 200)')
    args = parser.parse_args()

    success = 0
    failed = []

    tasks = []
    if args.dataset == 'mnist' or args.dataset is None:
        tasks.append(('MNIST', build_mnist_calib))
    if args.dataset == 'imagenet' or args.dataset is None:
        tasks.append(('ImageNet', build_imagenet_calib))

    for name, func in tasks:
        print(f"\n生成 {name} 校准数据 (N={args.num_samples})...")
        if func(args.num_samples):
            success += 1
        else:
            failed.append(name)

    print(f"\n校准数据生成完成: {success}/{len(tasks)} 成功")
    if failed:
        print(f"失败: {', '.join(failed)}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
