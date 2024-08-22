import torch
import cv2
import numpy as np


print(f"Torch version: {torch.__version__}")
print(f"Available devices: {torch.cuda.device_count()} GPU(s)")

# 加载MiDaS模型
model_type = "DPT_Large"
print("Loading MiDaS model...")
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

# 选择设备 (GPU 或 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
midas.to(device)

# 加载图像并转换为RGB格式
img_path = r"G:\real\1\fig.png"
print(f"Loading image from: {img_path}")
img = cv2.imread(img_path)

# 检查图像是否加载成功
if img is None:
    print("Error: Failed to load image. Check the file path.")
else:
    print(f"Image loaded successfully with shape: {img.shape}")

# 转换为RGB格式
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 加载MiDaS变换
print("Loading MiDaS transforms...")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# 根据模型类型选择适当的变换
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# 对图像进行预处理并将其转换为模型的输入格式
print("Transforming image...")
input_batch = transform(img).to(device)

# 打印输入张量的大小和数据类型
print(f"Input batch shape: {input_batch.shape}, dtype: {input_batch.dtype}")

# 禁用梯度计算，并生成深度图
with torch.no_grad():
    print("Running inference...")
    prediction = midas(input_batch)

    # 检查预测结果的大小和数据类型
    print(f"Prediction shape: {prediction.shape}, dtype: {prediction.dtype}")

    # 将深度图从张量转为numpy格式
    depth_map = prediction.squeeze().cpu().numpy()

# 检查深度图的最小值和最大值
print(f"Depth map stats - min: {depth_map.min()}, max: {depth_map.max()}")

# 将深度图归一化为0-255范围以进行可视化
depth_min = depth_map.min()
depth_max = depth_map.max()
depth_map = (255 * (depth_map - depth_min) / (depth_max - depth_min)).astype("uint8")

# 检查归一化后的深度图是否有意义的值
print(f"Normalized depth map stats - min: {depth_map.min()}, max: {depth_map.max()}")

# 显示或保存深度图
print("Displaying depth map...")
cv2.imshow('Depth Map', depth_map)
cv2.imwrite(r"G:\new_evaluation_data\2\depth2.png", depth_map)  # 保存为图像文件
cv2.waitKey(0)
cv2.destroyAllWindows()
