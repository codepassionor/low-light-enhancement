import numpy as np
import cv2
import torch
import os

class DiffusionModel:
    def __init__(self):
        # 初始化扩散模型
        pass

    def forward_process(self, z_t, t):
        """执行扩散模型的前向加噪过程"""
        sqrt_value = np.sqrt(1 - t)
        noisy_z_t = z_t + np.random.normal(0, 1, z_t.shape) * sqrt_value
        return noisy_z_t

    def reverse_process(self, z_t, t):
        """执行扩散模型的去噪过程"""
        denoised_z_t = z_t - 0.1 * t
        return denoised_z_t

# 初始化扩散模型
diffusion_model = DiffusionModel()

def linear_transform_noise(noise, mask, gamma=1.0, delta=0.0):
    transformed_noise = noise.copy()
    transformed_noise[mask > 0] = gamma * transformed_noise[mask > 0] + delta
    return transformed_noise

def to_me_similarity(x_i, x_j, m_i, eta=1.0):
    sim = np.dot(x_i, x_j)
    if m_i == 1:
        sim *= eta
    return sim

def to_me_merge(tokens, mask, eta=0.5):
    """应用Token Merging（ToMe）算法，基于相似性和前景标记合并token"""
    merged_tokens = []
    for i, token in enumerate(tokens):
        for j, other_token in enumerate(tokens):
            if i != j:
                sim = to_me_similarity(token, other_token, mask[i], eta)
                if sim > 0.5:
                    merged_token = (token + other_token) / 2
                    merged_tokens.append(merged_token)
                    break
        else:
            merged_tokens.append(token)
    return np.array(merged_tokens)

def object_centric_diffusion(image, mask, T=1000, N=20, gamma=0.25, phi=2.0, eta=0.5, delta=0.0, save_dir="output"):
    """
    实现对象中心采样过程，包含线性变换和ToMe。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 将mask形状扩展为与z_T相同的形状
    mask_expanded = np.expand_dims(mask, axis=-1)  # 形状变为 (1080, 1920, 1)
    mask_expanded = np.repeat(mask_expanded, 3, axis=-1)  # 形状变为 (1080, 1920, 3)

    # 初始化潜变量
    z_T = np.random.normal(0, 1, size=image.shape)  # 初始噪声

    # 分割前景和背景潜变量
    z_f_T = z_T * mask_expanded  # 前景潜变量
    z_b_T = z_T * (1 - mask_expanded)  # 背景潜变量

    delta_T = T // N
    Tb = int(gamma * T)

    # 对前景潜变量以正常速率进行采样
    for t in range(T, Tb, -delta_T):
        z_f_T = diffusion_model.forward_process(z_f_T, t / T)

    # 对背景潜变量以加速速率进行采样
    for t in range(T, Tb, -int(phi * delta_T)):
        z_b_T = diffusion_model.forward_process(z_b_T, t / T)

    # 归一化并合并前景和背景潜变量
    mean_zb = np.mean(z_b_T)
    std_zb = np.std(z_b_T)
    z_f_T_normalized = (z_f_T - mean_zb) / std_zb

    z_T_combined = z_f_T_normalized + z_b_T

    # 保存最终加噪的图片
    cv2.imwrite(os.path.join(save_dir, "noisy_image.png"), z_T_combined)

    # 应用线性变换噪声到前景区域
    noise = z_T_combined - image
    transformed_noise = linear_transform_noise(noise, mask_expanded, gamma, delta)
    z_T_combined = image + transformed_noise

    # 执行去噪过程前应用ToMe
    tokens = z_T_combined.reshape(-1, z_T_combined.shape[-1])  # 将图像展平为token序列
    final_tokens = to_me_merge(tokens, mask_expanded.flatten(), eta=eta)
    z_T_combined = final_tokens.reshape(z_T_combined.shape)  # 恢复到图像形状

    # 保存应用ToMe后的图片
    cv2.imwrite(os.path.join(save_dir, "pre_denoised_image.png"), z_T_combined)

    # 执行去噪过程
    final_image = diffusion_model.reverse_process(z_T_combined, T)

    # 保存去噪后的图片
    cv2.imwrite(os.path.join(save_dir, "denoised_image.png"), final_image)

    return final_image

# 使用
image = cv2.imread(r'G:\new_evaluation_data\2\fig.png')
mask = cv2.imread(r'G:\new_evaluation_data\2\mask.png', cv2.IMREAD_GRAYSCALE)

# 执行对象中心扩散过程
final_image = object_centric_diffusion(image, mask, save_dir=r'G:\new_evaluation_data\2\diffusion_output')
