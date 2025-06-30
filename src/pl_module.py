import lightning as L
import torch

from src.models.iresnet import iresnet18, iresnet34, iresnet50, iresnet100
from src.models.losses import ArcFace, CosFace, ElasticArcFace, ElasticCosFace

backbone_dict = {
    "iresnet18": iresnet18,
    "iresnet34": iresnet34,
    "iresnet50": iresnet50,
    "iresnet100": iresnet100,
}

header_dict = {
    "arcface": ArcFace,
    "cosface": CosFace,
    "elasticarc": ElasticArcFace,
    "elasticcos": ElasticCosFace,
}


class FembModule(L.LightningModule):
    def __init__(
        self,
        backbone: str = "iresnet50",
        embed_dim: int = 512,
        dropout: float = 0.0,
        header="arcface",
        n_classes: int = 10572,
        s: float = 64.0,
        m: float = 0.50,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        warmup_epoch: int = -1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone_dict[backbone](num_features=embed_dim, dropout=dropout)
        assert header in header_dict.keys()
        self.header = header_dict[header](
            in_features=embed_dim,
            out_features=n_classes,
            s=s,
            m=m,
        )

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(imgs)
        return feats

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        feats = self(imgs)
        ampl = torch.norm(feats, dim=1)
        max_ampl = torch.max(ampl)
        logits = self.header(feats, targets)
        # logits vector describes the probability for each image to belong to one of n_classes
        loss = self.criterion(logits, targets)
        optimizer_lr = self.optimizers().optimizer.param_groups[0]["lr"]
        self.log("loss", loss, prog_bar=True)
        self.log("optimizer_lr", optimizer_lr)
        self.log("max_ampl", max_ampl.item())
        return loss

    def configure_optimizers(self):
        def lr_step_func(epoch):
            return (
                ((epoch + 1) / (4 + 1)) ** 2
                if epoch < self.hparams.warmup_epoch
                else 0.1 ** len([m for m in [22, 30, 40] if m - 1 <= epoch])
            )

        optimizer = torch.optim.SGD(
            # Need to optimize over all parameters in the module!
            params=self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_step_func,
        )

        return optimizer, scheduler
