import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from cldm.cldm import ControlLDM
from omegaconf import OmegaConf
import os

os.environ["NO_PROXY"] = "huggingface.co"

def load_image_as_tensor(image_path, target_size=(64, 64)):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = torch.cat([image_tensor, torch.zeros_like(image_tensor[:, :1, :, :])], dim=1)
    return image_tensor

def show_image_from_tensor(tensor):
    tensor = tensor.squeeze(0)
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor + 1) / 2
    image = tensor.permute(1, 2, 0).cpu().detach().numpy()
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def run_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = OmegaConf.load("config.yaml")

    control_ldm = ControlLDM(
        control_stage_config=config.model.params.control_stage_config,
        control_key=config.model.params.control_key,
        only_mid_control=config.model.params.only_mid_control,
        first_stage_config=config.model.params.first_stage_config,
        cond_stage_config=config.model.params.cond_stage_config,
        unet_config=config.model.params.unet_config,
        timesteps=config.model.params.timesteps
    ).to(device)

    for name, param in control_ldm.control_model.named_parameters():
        if param.requires_grad:
            print(f"Layer {name}: Parameter mean: {param.mean().item()}")

    depth_image_path = r"G:\new_evaluation_data\2\depth.png"
    image_image_path = r"G:\new_evaluation_data\2\fig.png"

    depth_features = load_image_as_tensor(depth_image_path).to(device)
    image_features = load_image_as_tensor(image_image_path).to(device)

    print(f"Depth features min: {depth_features.min()}, max: {depth_features.max()}")
    print(f"Image features min: {image_features.min()}, max: {image_features.max()}")

    text_features = ["A sample prompt text"]
    text_embeds = control_ldm.get_learned_conditioning(text_features).to(device)
    print(f"Text embeds min: {text_embeds.min()}, max: {text_embeds.max()}")

    cond = {
        "c_concat": [depth_features, image_features],
        "c_crossattn": [text_embeds]
    }

    timesteps = torch.randint(0, 1000, (1,), device=device).long()
    noise = torch.randn_like(depth_features, device=device)
    print(f"Noise min: {noise.min()}, max: {noise.max()}")

    control = control_ldm.control_model(x=noise, hint=torch.cat(cond['c_concat'], 1), timesteps=timesteps, context=text_embeds)
    for i, c in enumerate(control):
        print(f"Control output at layer {i} min: {c.min()}, max: {c.max()}, mean: {c.mean()}")

    output = control_ldm.apply_model(noise, timesteps, cond)
    print(f"Output min: {output.min()}, max: {output.max()}")

    if output.max() - output.min() == 0:
        print("Output contains only a single value, resulting in a blank image.")
    else:
        show_image_from_tensor(output)

run_model()
