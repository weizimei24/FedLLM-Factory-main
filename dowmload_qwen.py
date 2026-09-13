from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-1.7B",
    local_dir="./Qwen3-1.7B",
)

print("下载完成")