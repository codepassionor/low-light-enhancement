import torch
from diffusers import UNet2DConditionModel
from utils.text_embedding import generate_text_embeddings

class DistFeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = []
        self.hooks = []
        self.register_hook()

    def hook_function(self, module, input, output):
        # No need to gather output from different devices; we just store the output
        self.features.append(output)

    def register_hook(self):
        # Adjust layers as per your architecture
        res_dict = {1: [1]}  # Example configuration
        unet = self.model
        for res in res_dict:
            for block in res_dict[res]:
                module = unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
                self.hooks.append(module.register_forward_hook(self.hook_function))

    def get_features(self, input_data):
        self.reset()  # Ensure features are empty before running the model
        _ = self.model(input_data)
        return self.features

    def reset(self):
        self.features = []

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
