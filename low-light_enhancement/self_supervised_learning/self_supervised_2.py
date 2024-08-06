from pytorch_lightning import LightningModule, Trainer

class SimCLR(LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        resnet.fc = torch.nn.Identity()
        self.backbone = resnet
        self.projection_head = heads.SimCLRProjectionHead(512, 512, 128)
        self.criterion = loss.NTXentLoss()

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z

    def training_step(self, batch, batch_index):
        (view0, view1), _, _ = batch
        z0 = self.forward(view0)
        z1 = self.forward(view1)
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim


model = SimCLR()
trainer = Trainer(max_epochs=10, devices=1, accelerator="gpu")
trainer.fit(model, dataloader)



# # Use distributed version of loss functions.
# criterion = loss.NTXentLoss(gather_distributed=True)

# trainer = Trainer(
#     max_epochs=10,
#     devices=4,
#     accelerator="gpu",
#     strategy="ddp",
#     sync_batchnorm=True,
#     use_distributed_sampler=True,  # or replace_sampler_ddp=True for PyTorch Lightning <2.0
# )
# trainer.fit(model, dataloader)
# training on four gpu