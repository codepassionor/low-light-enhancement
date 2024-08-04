from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    LCMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from utils.text_embedding import generate_text_embeddings
import torch
class DistFeatureExtractor:
    def __init__(self, model, accelerator):
        self.model = model
        self.features = []
        self.hooks = []
        self.register_hook()
        self.accelerator = accelerator

    def hook_function(self, module, input, output):
        output_gathered = self.accelerator.gather(output)
        if self.accelerator.is_main_process:
            print('output_gathered.shape')
            print(output_gathered.shape)
            self.features.append(output_gathered)

    def register_hook(self):
        #print(self.model._modules)
        res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
        unet = self.model
        hooks = []
        for res in res_dict:
            for block in res_dict[res]:
                module = unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
                self.hooks.append(module.register_forward_hook(self.hook_function))

    def get_features(self, input_data):
        _ = self.model(input_data)
        return self.features

    def reset(self):
        self.features = []

    def remove_hook(self):
        self.hook.remove()


class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = []
        self.hooks = []
        self.register_hook()

    def hook_function(self, module, input, output):
        self.features.append(output)

    def register_hook(self):
        #print(self.model._modules)
        res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
        unet = self.model
        hooks = []
        for res in res_dict:
            for block in res_dict[res]:
                module = unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
                self.hooks.append(module.register_forward_hook(self.hook_function))

    def get_features(self, input_data):
        _ = self.model(input_data)
        return self.features

    def reset(self):
        self.features = []

    def remove_hook(self):
        self.hook.remove()

if __name__ == '__main__':
    num_steps = 1000
    load_model = UNet2DConditionModel.from_pretrained('/data/workspace/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9', subfolder="unet").cuda()
    feature_extractor = FeatureExtractor(load_model)
    text = ''
    text_embeddings = generate_text_embeddings(text)
    data = torch.zeros((1,4, 64, 64)).cuda()
    t = torch.randint(0, num_steps, (1,)).long().cuda()
    features = feature_extractor.get_features((data, t, text_embeddings))
    for idx, feature in enumerate(features):
        print(f"Feature map {idx} shape: {feature.shape}")

    # Clean up hooks when done
    feature_extractor.remove_hooks()