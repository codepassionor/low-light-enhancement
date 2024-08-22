import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
from cldm.ddim_hacked import DDIMSampler
from torchvision.transforms.functional import resize as tv_resize, crop as tv_crop
import yaml
from ldm.util import instantiate_from_config
import torch.hub

# 临时禁用 Hugging Face 的代理设置
os.environ["NO_PROXY"] = "huggingface.co"

# 准备深度图的函数，使用 MiDaS 生成深度图
def prepare_depth_maps(image, model_type='DPT_Large', device='cuda'):
    midas = torch.hub.load("intel-isl/MiDaS", model_type, force_reload=False)
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", force_reload=False)
    transform = midas_transforms.dpt_transform if model_type in ["DPT_Large",
                                                                 "DPT_Hybrid"] else midas_transforms.small_transform

    if isinstance(image, torch.Tensor):
        image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()

    img_tensor = transform(image).to(device)
    prediction = midas(img_tensor)

    # 确保深度图的形状正确
    print(f"Generated depth map shape: {prediction.shape}")

    depth_map = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=(image.shape[0] // 8, image.shape[1] // 8),
        mode="bicubic",
        align_corners=False,
    )

    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0

    return depth_map.to(torch.float16).to(device)

# 中心扩散的采样函数
def object_centric_diffusion(image, mask, control_ldm, ddim_sampler, latents_path, bevdepth_model=None,
                             timesteps=500, steps_per_region=10, gamma=0.8, phi=1.2, device='cuda'):
    image = image.to(device).float()

    # 检查 mask 的形状
    print(f"Original mask shape: {mask.shape}")

    if len(mask.shape) == 4:
        mask = mask.squeeze(0).unsqueeze(1).to(device).float()

    print(f"Reshaped mask shape: {mask.shape}")

    with torch.no_grad():
        if bevdepth_model is not None:
            image_resized = tv_resize(image, (256, 704))
            image_cropped = tv_crop(image_resized, 140, 0, 256, 704).unsqueeze(0)
            bevdepth_output = bevdepth_model(image_cropped)
            depth_map = bevdepth_output['depth']
        else:
            depth_map = prepare_depth_maps(image, device=device)

    # 确保 z_T 和 mask 及 depth_map 的形状匹配
    z_T = torch.randn((image.shape[0], 4, image.shape[2] // 8, image.shape[3] // 8)).to(device)
    mask = F.interpolate(mask, size=z_T.shape[2:], mode='nearest')
    depth_map = F.interpolate(depth_map, size=z_T.shape[2:], mode='nearest')

    # 检查生成的 encoder_hidden_states 和 depth_map 的形状
    text_input = ["This is a sample description"] * image.shape[0]
    encoder_hidden_states = control_ldm.get_learned_conditioning(text_input)
    print(f"Shape of encoder_hidden_states: {encoder_hidden_states.shape}")
    print(f"Shape of depth_map: {depth_map.shape}")

    cond = {"c_concat": [depth_map], "c_crossattn": [encoder_hidden_states]}

    z_foreground = z_T * mask
    for t in range(timesteps - 1, int(gamma * timesteps) - 1, -timesteps // steps_per_region):
        z_foreground = ddim_sampler.p_sample_ddim(z_foreground, cond, t)

    z_background = torch.randn_like(z_foreground).to(device)
    for t in range(timesteps - 1, int(gamma * timesteps) - 1, -int(phi * (timesteps // steps_per_region))):
        z_background = ddim_sampler.p_sample_ddim(z_background, cond, t)

    z_combined = z_foreground + z_background

    for t in range(int(gamma * timesteps) - 1, -1, -timesteps // steps_per_region):
        z_combined = ddim_sampler.p_sample_ddim(z_combined, cond, t)

    z_combined = 1 / 0.18215 * z_combined
    final_image = control_ldm.first_stage_model.decode(z_combined).sample
    final_image = (final_image / 2 + 0.5).clamp(0, 1)

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

# 执行扩散过程
output_dir = "path/to/output"
os.makedirs(output_dir, exist_ok=True)
final_image = object_centric_diffusion(image_tensor, mask_tensor, control_ldm, ddim_sampler, latents_path=output_dir,
                                       device=device)
