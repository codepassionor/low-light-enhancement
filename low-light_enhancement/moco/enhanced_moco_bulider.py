import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    build a MoCo model with an encoder_q, an encoder_k, and a queue
    with Dataset discrimination
    """
    def __init__(self,
                 base_encoder,
                 dim=128,
                 K=8192,
                 m=0.999,
                 T=0.07,
                 mlp=False):
        super().__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoder
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim),
                nn.ReLU(),
                self.encoder_q.fc
            )

            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                self.encoder_k.fc
            )

        # initialize the params
        for param_q, param_k in zip(
                self.encoder_q.parameters, self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # create the queue and pointer
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # 入队 出队的字段
        bs = keys.shape[0]
        ptr = int(self.queue_ptr)

        assert self.K % bs == 0

        self.queue[:, ptr: ptr + bs] = keys.T

        ptr = (ptr + bs) % self.K

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def momentum_update_encoder_k(self):
        """
        momentum update the key encoder
        keep in mind necessary
        """

        # use for loop
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

    def forward(self, imgs1, imgs2):
        """
        forward method of Dataset Discrimination
        with multi-kinds images,
        imgs1 imgs2 are lists of auged img tensors
        """

        for i in range(len(imgs1)):
            imgs1[i] = self.encoder_q(imgs1[i])

        with torch.no_grad():
            self.momentum_update_encoder_k()
            for i in range(len(imgs2)):
                imgs2[i] = self.encoder_k(imgs2[i])

        # with shape of N*nC where n=len(imgs1)
        query = torch.cat(imgs1, dim=1)
        key_pos = torch.cat(imgs2, dim=1)

        # normalization
        query = nn.functional.normalize(query, dim=1)
        key_pos = nn.functional.normalize(key_pos, dim=1)

        # calc the pos logits  N*1
        l_pos = torch.einsum("nNc, nNc->nNN", [query, key_pos]).mean(dim=2).reshape(query.shape[0], -1).mean(dim=1)

        # calc the neg logits N*K
        l_neg = torch.einsum("nNc, ncK->nNK", [query, self.queue.unsqueeze(0).expand(len(imgs1), -1, -1)]).mean(dim=0)

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        self._dequeue_and_enqueue(key_pos)

        return logits, labels




