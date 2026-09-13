from modelscope import snapshot_download

model_dir = snapshot_download(
    "Qwen/Qwen3-1.7B",
    local_dir="./Qwen3-1.7B"
)

print("模型路径:", model_dir)