import torch
import open_clip

# 检查点路径
checkpoint_path = r"C:\Users\Administrator\.cache\huggingface\hub\models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K\resolve\main\open_clip_pytorch_model.bin"

# 加载 CLIP 模型
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-H-14", pretrained=None, device='cpu'
)

# 加载预训练的权重
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint)
print("CLIP model checkpoint loaded successfully!")

# 生成随机输入进行推理测试
dummy_input = torch.randn(1, 3, 224, 224).to('cpu')  # CLIP 使用 224x224 的图像尺寸
model.eval()

# 执行推理
with torch.no_grad():
    output = model.encode_image(dummy_input)

print(f"Output shape: {output.shape}, min: {output.min()}, max: {output.max()}")
