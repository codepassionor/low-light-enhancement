import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from utils.text_embedding import generate_text_embeddings
from hook_features import FeatureExtractor
from unet.get_unet_parameters import get_upsample_64x64_params

class MoCo(nn.Module):
    def __init__(self,
                 base_encoder,
                 dim=128,
                 K=8192,
                 m=0.999,
                 T=0.07,
                 mlp=False):
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # Create the encoders (query and key)
        self.encoder_q = base_encoder
        self.encoder_k = base_encoder

        # Initialize the parameters of encoder_k to be the same as encoder_q
        for param_q, param_k in zip(
                get_upsample_64x64_params(self.encoder_q), get_upsample_64x64_params(self.encoder_k)
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create the queue and initialize the pointer
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, "K should be divisible by batch size"

        # Dequeue and enqueue the keys
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def momentum_update_encoder_k(self):
        # Momentum update of encoder_k
        for param_q, param_k in zip(
            get_upsample_64x64_params(self.encoder_q), get_upsample_64x64_params(self.encoder_k)
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

    def forward(self, imgs1, imgs2):
        query_features = []
        key_features = []

        # Extract features from imgs1 using encoder_q
        feature_extractor_q = FeatureExtractor(self.encoder_q)
        for img in imgs1:
            _ = feature_extractor_q.get_features(img)
            query_features.append(feature_extractor_q.features[-1])  # Assuming last feature map is 64x64

        # Extract features from imgs2 using encoder_k
        feature_extractor_k = FeatureExtractor(self.encoder_k)
        with torch.no_grad():
            self.momentum_update_encoder_k()
            for img in imgs2:
                _ = feature_extractor_k.get_features(img)
                key_features.append(feature_extractor_k.features[-1])  # Assuming last feature map is 64x64

        # Concatenate the query and key features
        query = torch.cat(query_features, dim=1)
        key_pos = torch.cat(key_features, dim=1)

        # Normalize the features
        query = nn.functional.normalize(query, dim=1)
        key_pos = nn.functional.normalize(key_pos, dim=1)

        # Calculate positive logits (N*1)
        l_pos = torch.einsum("nc,nc->n", [query, key_pos]).unsqueeze(-1)

        # Calculate negative logits (N*K)
        l_neg = torch.einsum("nc,ck->nk", [query, self.queue.clone().detach()])

        # Concatenate positive and negative logits
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        # Labels: positive keys are the first in the batch
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # Dequeue and enqueue the keys
        self._dequeue_and_enqueue(key_pos)

        return logits, labels
