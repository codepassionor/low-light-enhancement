import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
from ldm.models.diffusion.ddim import DDIMSampler
from torchvision.transforms.functional import resize as tv_resize, crop as tv_crop
import yaml
from ldm.util import instantiate_from_config
from bevdepth.exps.nuscenes.base_exp import BEVDepthLightningModel

# 中心扩散的采样函数
def object_centric_diffusion(image, mask, control_ldm, ddim_sampler, bevdepth_model, latents_path,
                             timesteps=500, steps_per_region=10, gamma=0.8, phi=1.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device).float()
    mask = mask.unsqueeze(0).unsqueeze(0).to(device).float()

    # 使用 BEVDepth 生成深度图
    with torch.no_grad():
        # 调整图像尺寸以适应 BEVDepth 模型
        image_resized = tv_resize(image, (256, 704))
        image_cropped = tv_crop(image_resized, 140, 0, 256, 704).unsqueeze(0)

        # 输入到 BEVDepth 模型
        bevdepth_output = bevdepth_model(image_cropped)
        depth_map = bevdepth_output['depth']  #  BEVDepth 模型输出的深度图

    # 初始化潜变量
    z_T = torch.randn((image.shape[0], 4, image.shape[2] // 8, image.shape[3] // 8)).to(device)
    mask = F.interpolate(mask, size=z_T.shape[2:], mode='nearest')
    depth_map = F.interpolate(depth_map, size=z_T.shape[2:], mode='nearest')

    # 提取文本编码
    encoder_hidden_states = control_ldm.get_unconditional_conditioning(image.shape[0])
    cond = {"c_concat": [depth_map], "c_crossattn": [encoder_hidden_states]}

    # 前景采样
    z_foreground = z_T * mask
    for t in range(timesteps - 1, int(gamma * timesteps) - 1, -timesteps // steps_per_region):
        z_foreground = ddim_sampler.p_sample_ddim(z_foreground, cond, t)

    # 背景采样
    z_background = torch.randn_like(z_foreground).to(device)
    for t in range(timesteps - 1, int(gamma * timesteps) - 1, -int(phi * (timesteps // steps_per_region))):
        z_background = ddim_sampler.p_sample_ddim(z_background, cond, t)

    # 合并前景和背景
    z_combined = z_foreground + z_background

    # 去噪过程
    for t in range(int(gamma * timesteps) - 1, -1, -timesteps // steps_per_region):
        z_combined = ddim_sampler.p_sample_ddim(z_combined, cond, t)

    # 解码
    z_combined = 1 / 0.18215 * z_combined
    final_image = control_ldm.first_stage_model.decode(z_combined).sample
    final_image = (final_image / 2 + 0.5).clamp(0, 1)

    # 保存图像
    for i, img in enumerate(final_image):
        img = (img.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(latents_path, f"denoised_image_{i}.png"))

    return final_image

# 加载模型和采样器
def load_models(config_path, device="cuda"):
    # 读取 YAML 配置文件
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 使用配置文件实例化模型并自动加载 Hugging Face 上的权重
    control_ldm = instantiate_from_config(config['model'])
    control_ldm = control_ldm.to(device).eval()

    ddim_sampler = DDIMSampler(control_ldm)
    return control_ldm, ddim_sampler

# 主要流程
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 你的图片和掩码已经加载
image_path = r'G:\new_evaluation_data\2\fig.png'
mask_path = r'G:\new_evaluation_data\2\mask.png'
image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")

# 图像预处理
image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255
mask_tensor = torch.tensor(np.array(mask)).unsqueeze(0).unsqueeze(0).float() / 255

# 加载模型和采样器
config_path = r"C:\Users\Administrator\Downloads\LightDiff-main\models\cldm_v21.yaml"
control_ldm, ddim_sampler = load_models(config_path, device)

# 直接使用 BEVDepthLightningModel 实例化模型
bevdepth_model = BEVDepthLightningModel()

# 执行扩散过程
output_dir = "path/to/output"
os.makedirs(output_dir, exist_ok=True)
final_image = object_centric_diffusion(image_tensor, mask_tensor, control_ldm, ddim_sampler, bevdepth_model, output_dir)
