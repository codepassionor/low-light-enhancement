import numpy as np
import cv2
import torch
import os
from tokenflow import TokenFlow
import yaml
from PIL import Image
import torchvision.transforms as T
from transformers import CLIPTokenizer


def save_noisy_latents(latents, t, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    latents_path = os.path.join(save_dir, f"noisy_latents_{t}.pt")
    torch.save(latents, latents_path)


def object_centric_diffusion(image, mask, editor, latents_path, T=1000, N=20, gamma=0.25, phi=2.0, eta=0.5, delta=0.0):
    # 创建存储潜变量的目录
    if not os.path.exists(latents_path):
        os.makedirs(latents_path)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确保 mask 和 image 都是 float32 并放在同一个设备上
    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).to(device).float()  # 将 mask 转换为单通道
    image = image.to(device).float()

    # 初始化潜变量 z_T
    z_T = torch.randn_like(image).float().to(device)

    # 确保 `editor.eps` 在正确的设备上
    if editor.eps is None or editor.eps.shape != z_T.shape:
        editor.eps = torch.randn_like(z_T).to(device).float()

    # 分离前景和背景潜变量
    z_f_T = z_T * mask
    z_b_T = z_T * (1 - mask)

    # 如果前景和背景潜变量的通道数是 3，添加一个额外的全零通道，确保通道数为 4
    if z_f_T.shape[1] == 3:
        zero_channel = torch.zeros_like(z_f_T[:, :1, :, :])  # 创建一个全零的通道
        z_f_T = torch.cat([z_f_T, zero_channel], dim=1)  # 添加到前景潜变量

    if z_b_T.shape[1] == 3:
        zero_channel = torch.zeros_like(z_b_T[:, :1, :, :])  # 创建一个全零的通道
        z_b_T = torch.cat([z_b_T, zero_channel], dim=1)  # 添加到背景潜变量

    # 调整 T 不超过 timesteps 的大小
    max_T = len(editor.scheduler.timesteps)
    T = min(T, max_T)  # 确保 T 不超过调度器的步数大小
    delta_T = T // N
    Tb = int(gamma * T)

    # 假设您的 editor 中有一个 text_encoder 生成 encoder_hidden_states
    prompt_text = "A detailed description of your scene"  # 您的文本提示

    # 使用 CLIPTokenizer 将文本转换为 input_ids
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    # 获取 encoder_hidden_states
    encoder_hidden_states = editor.text_encoder(**inputs).last_hidden_state.to(device)

    # 前景正常速率采样
    for t in range(T - 1, Tb - 1, -delta_T):
        # 使用 UNet 模型预测前景潜变量的输出，并传入 encoder_hidden_states
        model_output = editor.unet(z_f_T, t, encoder_hidden_states=encoder_hidden_states)
        z_f_T = editor.scheduler.step(model_output=model_output, timestep=editor.scheduler.timesteps[t],
                                      sample=z_f_T).prev_sample

    # 背景加速采样
    for t in range(T - 1, Tb - 1, -int(phi * delta_T)):
        # 使用 UNet 模型预测背景潜变量的输出，并传入 encoder_hidden_states
        model_output = editor.unet(z_b_T, t, encoder_hidden_states=encoder_hidden_states)
        z_b_T = editor.scheduler.step(model_output=model_output, timestep=editor.scheduler.timesteps[t],
                                      sample=z_b_T).prev_sample

    # 归一化并合并前景和背景潜变量
    mean_zb = torch.mean(z_b_T)
    std_zb = torch.std(z_b_T)
    z_f_T_normalized = (z_f_T - mean_zb) / std_zb
    z_T_combined = z_f_T_normalized + z_b_T

    # 合并后继续去噪采样
    for t in range(Tb - 1, -1, -delta_T):
        # 使用 UNet 模型预测合并后的潜变量的输出，并传入 encoder_hidden_states
        model_output = editor.unet(z_T_combined, t, encoder_hidden_states=encoder_hidden_states)
        z_T_combined = editor.scheduler.step(model_output=model_output, timestep=editor.scheduler.timesteps[t],
                                             sample=z_T_combined).prev_sample

    # 检查 z_T_combined 的通道数，如果只有 3 个通道，添加一个额外的通道
    if z_T_combined.shape[1] == 3:
        zero_channel = torch.zeros_like(z_T_combined[:, :1, :, :])  # 创建一个全零的通道
        z_T_combined = torch.cat([z_T_combined, zero_channel], dim=1)

    # 确保输入和VAE权重的精度匹配
    z_T_combined = z_T_combined.to(dtype=editor.vae.post_quant_conv.weight.dtype)

    # 解码最终图像
    final_image = editor.decode_latents(z_T_combined)

    # 保存去噪后的图像
    for i, img in enumerate(final_image):
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
        cv2.imwrite(os.path.join(latents_path, f"denoised_image_{i}.png"), img.cpu().numpy().astype(np.uint8))

    return final_image


# 处理单张图片
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_path = r'G:\new_evaluation_data\2\fig.png'
mask_path = r'G:\new_evaluation_data\2\mask.png'

image = Image.open(image_path).convert('RGB')
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

image_tensor = T.ToTensor()(image).unsqueeze(0)

with open('config_pnp.yaml', "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

latents_dir = r'G:\new_evaluation_data\2\diffusion_output'
os.makedirs(latents_dir, exist_ok=True)
latents_path = os.path.join(latents_dir, 'latents.pt')

if not os.path.exists(latents_path):
    editor_temp = TokenFlow(config)
    latents = editor_temp.encode_imgs(image_tensor.to(editor_temp.device))
    torch.save(latents, latents_path)
else:
    print(f"Latents already exist at: {latents_path}")

editor = TokenFlow(config)

# 确保editor模型内部转换为float32
editor.unet.to(device).float()

final_image = object_centric_diffusion(image_tensor, mask, editor, latents_path)
