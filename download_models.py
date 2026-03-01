"""
模型下载脚本

使用 gdown 从 Google Drive 下载预训练模型权重.
支持命令行选择下载哪些模型.
"""

import os
import sys
import argparse
from pathlib import Path

# Model download info
# Google Drive file IDs (from shared folders)
MODELS = {
    "flowformer": {
        "things": {
            "file_id": None,  # 需要从 Google Drive 获取具体 file ID
            "url": "https://drive.google.com/drive/folders/1K2dcWxaqOLiQ3PoqRdokrgWsGIf3yBA_",
            "save_path": "checkpoints/flowformer/things.pth",
            "description": "FlowFormer - Things3D (推荐, 泛化性好)",
        },
        "sintel": {
            "file_id": None,
            "url": "https://drive.google.com/drive/folders/1K2dcWxaqOLiQ3PoqRdokrgWsGIf3yBA_",
            "save_path": "checkpoints/flowformer/sintel.pth",
            "description": "FlowFormer - Sintel",
        },
        "kitti": {
            "file_id": None,
            "url": "https://drive.google.com/drive/folders/1K2dcWxaqOLiQ3PoqRdokrgWsGIf3yBA_",
            "save_path": "checkpoints/flowformer/kitti.pth",
            "description": "FlowFormer - KITTI",
        },
    },
    "flowformerpp": {
        "things": {
            "file_id": None,
            "url": "https://drive.google.com/drive/folders/1fyPZvcH4SuNCgnBvIJB2PktT5IN9PYPI",
            "save_path": "checkpoints/flowformerpp/things.pth",
            "description": "FlowFormer++ - Things3D (推荐, 泛化性好)",
        },
        "sintel": {
            "file_id": None,
            "url": "https://drive.google.com/drive/folders/1fyPZvcH4SuNCgnBvIJB2PktT5IN9PYPI",
            "save_path": "checkpoints/flowformerpp/sintel.pth",
            "description": "FlowFormer++ - Sintel",
        },
        "kitti": {
            "file_id": None,
            "url": "https://drive.google.com/drive/folders/1fyPZvcH4SuNCgnBvIJB2PktT5IN9PYPI",
            "save_path": "checkpoints/flowformerpp/kitti.pth",
            "description": "FlowFormer++ - KITTI",
        },
    },
}


def list_available_models():
    """列出所有可下载的模型"""
    print("\n可下载的模型列表:")
    print("=" * 60)
    for algo, models in MODELS.items():
        print(f"\n[{algo}]")
        for name, info in models.items():
            status = "✅ 已存在" if Path(info["save_path"]).exists() else "❌ 未下载"
            print(f"  {name:12s} - {info['description']}")
            print(f"               路径: {info['save_path']}  {status}")
    print()


def download_model(algo: str, model_name: str, force: bool = False):
    """
    下载指定模型.

    由于 Google Drive 共享文件夹需要具体的 file_id,
    如果 file_id 未设置, 将提示用户手动下载.
    """
    try:
        import gdown
    except ImportError:
        print("请先安装 gdown: pip install gdown")
        sys.exit(1)

    if algo not in MODELS:
        print(f"未知算法: {algo}. 可选: {list(MODELS.keys())}")
        return

    if model_name not in MODELS[algo]:
        print(f"未知模型: {model_name}. 可选: {list(MODELS[algo].keys())}")
        return

    info = MODELS[algo][model_name]
    save_path = Path(info["save_path"])

    if save_path.exists() and not force:
        print(f"模型已存在: {save_path} (使用 --force 覆盖)")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    if info["file_id"]:
        # 使用 gdown 下载
        url = f"https://drive.google.com/uc?id={info['file_id']}"
        print(f"正在下载 {algo}/{model_name} ...")
        gdown.download(url, str(save_path), quiet=False)
        print(f"下载完成: {save_path}")
    else:
        # file_id 未知, 提示手动下载
        print(f"\n[手动下载指引] {algo}/{model_name}")
        print(f"  1. 访问: {info['url']}")
        print(f"  2. 下载文件: {model_name}.pth")
        print(f"  3. 放置到: {save_path.resolve()}")
        print()


def check_models(config: dict) -> bool:
    """
    检查配置中所需的模型是否存在.

    Returns:
        True 如果所有模型都存在
    """
    all_ok = True
    for algo in config.get("algorithms", []):
        ckpt = config.get(algo, {}).get("checkpoint", "")
        if ckpt and not Path(ckpt).exists():
            print(f"[WARNING] 模型不存在: {ckpt}")
            print(f"  请运行 'python download_models.py --list' 查看下载指引")
            all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="下载预训练模型")
    parser.add_argument("--list", action="store_true", help="列出所有可用模型")
    parser.add_argument("--algo", type=str, choices=["flowformer", "flowformerpp"],
                        help="算法名称")
    parser.add_argument("--model", type=str, default="things",
                        help="模型名称 (默认: things)")
    parser.add_argument("--all", action="store_true", help="下载所有模型")
    parser.add_argument("--force", action="store_true", help="强制覆盖已存在的模型")

    args = parser.parse_args()

    if args.list:
        list_available_models()
        return

    if args.all:
        for algo in MODELS:
            for model_name in MODELS[algo]:
                download_model(algo, model_name, force=args.force)
        return

    if args.algo:
        download_model(args.algo, args.model, force=args.force)
    else:
        # 默认下载两个算法的 things 模型
        print("默认下载 things.pth (两个算法各一个)...")
        download_model("flowformer", "things", force=args.force)
        download_model("flowformerpp", "things", force=args.force)


if __name__ == "__main__":
    main()
