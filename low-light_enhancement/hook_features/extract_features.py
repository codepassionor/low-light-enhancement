import torch
from PIL import Image
from diffusers import UNet2DConditionModel
from utils.text_embedding import generate_text_embeddings
from hook_features import FeatureExtractor
from torchvision import transforms

def extract_features(model_path, text, image_path, num_steps=1000, device='cuda'):
    # 加载模型
    load_model = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(device)
    feature_extractor = FeatureExtractor(load_model)

    # 生成文本嵌入
    text_embeddings = generate_text_embeddings(text).to(device)

    # 读取并预处理图像
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 根据模型输入大小调整
        transforms.ToTensor()
    ])
    data = transform(image).unsqueeze(0).to(device)  # 增加批次维度

    # 生成随机时间步长
    t = torch.randint(0, num_steps, (1,)).long().to(device)

    # 提取特征
    features = feature_extractor.get_features((data, t, text_embeddings))

    # 打印特征图的形状
    for idx, feature in enumerate(features):
        print(f"Feature map {idx} shape: {feature.shape}")

    # 移除钩子
    feature_extractor.remove_hooks()
    return features

if __name__ == '__main__':
    model_path = '/data/workspace/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9'
    text = 'A photo of a beautiful landscape'
    image_path = '/path/to/your/image.jpg'  # 替换为实际的图像路径
    features = extract_features(model_path, text, image_path)
