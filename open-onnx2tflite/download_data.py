"""
数据集下载脚本

下载 MNIST 和 Imagenette 320px 数据集，用于精度验证与 INT8 量化校准。

使用方法:
    python download_data.py              # 下载全部数据集
    python download_data.py --dataset mnist        # 仅下载 MNIST
    python download_data.py --dataset imagenette   # 仅下载 Imagenette
"""

import argparse
import gzip
import os
import subprocess
import sys


DATA_DIR = 'data'

# MNIST 文件列表（AWS 公开镜像）
MNIST_FILES = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz',
]
MNIST_BASE_URL = 'https://ossci-datasets.s3.amazonaws.com/mnist'

# Imagenette 320px
IMAGENETTE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'
IMAGENETTE_FILE = 'imagenette2-320.tgz'
IMAGENETTE_DIR = 'imagenette2-320'


def download_mnist():
    """下载 MNIST 数据集"""
    dest_dir = os.path.join(DATA_DIR, 'mnist')
    os.makedirs(dest_dir, exist_ok=True)

    print("下载 MNIST 数据集...")
    all_exist = True
    for f in MNIST_FILES:
        if not os.path.exists(os.path.join(dest_dir, f)):
            all_exist = False
            break

    if all_exist:
        print("  [跳过] MNIST 文件已存在")
        return True

    for f in MNIST_FILES:
        dest = os.path.join(dest_dir, f)
        if os.path.exists(dest):
            print(f"  [跳过] {f}")
            continue

        url = f"{MNIST_BASE_URL}/{f}"
        print(f"  下载 {f}...")
        result = subprocess.run(
            ['wget', '-q', url, '-O', dest],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  [失败] {f}: {result.stderr.strip()}")
            return False
        print(f"  [完成] {f}")

    # 解压 .gz 文件
    print("  解压 MNIST 文件...")
    for f in MNIST_FILES:
        gz_path = os.path.join(dest_dir, f)
        out_path = os.path.join(dest_dir, f[:-3])  # 去掉 .gz
        if os.path.exists(out_path):
            continue
        with gzip.open(gz_path, 'rb') as fin:
            with open(out_path, 'wb') as fout:
                fout.write(fin.read())
        print(f"  [解压] {f[:-3]}")

    print("MNIST 下载完成")
    return True


def download_imagenette():
    """下载 Imagenette 320px 数据集"""
    dest_dir = DATA_DIR
    os.makedirs(dest_dir, exist_ok=True)

    # 检查是否已解压
    extracted_dir = os.path.join(dest_dir, IMAGENETTE_DIR)
    if os.path.exists(extracted_dir):
        print("  [跳过] Imagenette 已解压存在")
        return True

    tgz_path = os.path.join(dest_dir, IMAGENETTE_FILE)

    # 下载 tgz
    if not os.path.exists(tgz_path):
        print(f"下载 Imagenette 320px (~350 MB)...")
        result = subprocess.run(
            ['wget', '--show-progress', IMAGENETTE_URL, '-O', tgz_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  [失败] Imagenette 下载失败")
            if os.path.exists(tgz_path):
                os.remove(tgz_path)
            return False
        print("  [完成] Imagenette 下载")
    else:
        print("  [跳过] Imagenette tgz 已存在")

    # 解压
    print("  解压 Imagenette...")
    result = subprocess.run(
        ['tar', '-xzf', tgz_path, '-C', dest_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [失败] 解压失败: {result.stderr.strip()}")
        return False
    print("  [完成] Imagenette 解压")

    print("Imagenette 下载完成")
    return True


def main():
    parser = argparse.ArgumentParser(description='下载数据集')
    parser.add_argument('--dataset', type=str,
                        choices=['mnist', 'imagenette'],
                        help='指定数据集')
    args = parser.parse_args()

    success = 0
    failed = []

    tasks = []
    if args.dataset == 'mnist' or args.dataset is None:
        tasks.append(('MNIST', download_mnist))
    if args.dataset == 'imagenette' or args.dataset is None:
        tasks.append(('Imagenette', download_imagenette))

    for name, func in tasks:
        print()
        if func():
            success += 1
        else:
            failed.append(name)

    print(f"\n下载完成: {success}/{len(tasks)} 成功")
    if failed:
        print(f"失败: {', '.join(failed)}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
