# 运行前请确保已安装依赖: pip install huggingface_hub
import os
import shutil
from huggingface_hub import hf_hub_download

def download_dataset():
    # 你的专属 Hugging Face 仓库
    repo_id = "LW0324/FinRL-Ashare-Stock-Data"
    filename = "tensor_data_raw.npz"
    local_path = "./tensor_data_raw.npz"

    if os.path.exists(local_path):
        print(f"✅ 检测到本地已存在数据集: {local_path}，跳过下载。")
        return

    print(f"📥 正在从 Hugging Face 下载数据集: {filename}...")
    print(f"🔗 目标仓库: https://huggingface.co/datasets/{repo_id}")
    print("⏳ 文件约为 1GB，具备断点续传功能，请耐心等待...")

    try:
        # hf_hub_download 会自动处理下载和本地缓存
        cached_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset"
        )

        # 将缓存目录中的文件复制到当前工作目录
        shutil.copy(cached_file, local_path)
        print(f"🎉 数据集下载成功！已保存至: {os.path.abspath(local_path)}")

    except Exception as e:
        print(f"❌ 下载失败！请检查网络或仓库名称是否正确。错误信息: {e}")

if __name__ == "__main__":
    download_dataset()