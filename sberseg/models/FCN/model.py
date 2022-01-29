import torch
import torchvision
import pytorch_lightning as pl
from torch.nn import functional as F

class FCN(pl.LightningModule):
    def __init__(self):
        super(FCN, self).__init__()
        self.learning_rate = 1e-3
        self.net = torchvision.models.segmentation.fcn_resnet50(
            pretrained = False, 
            progress = True, 
            num_classes = 8
        )
        # self.net = UNet(num_classes = 19, bilinear = False)
        # self.net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained = False, progress = True, num_classes = 19)
        # self.net = ENet(num_classes = 19)
        
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_nb) :
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)['out']
        loss_val = F.cross_entropy(out, mask, ignore_index = 250)
        return {'loss' : loss_val}
    
    def training_step_end(self, outs):
        self.log("train/loss", outs['loss'])

    def validation_step(self, batch, batch_nb) :
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)['out']

        loss_val = F.cross_entropy(out, mask, ignore_index = 250)
        return {'loss' : loss_val}
    
    def validation_step_end(self, outs):
        self.log("val/loss", outs['loss'])
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
        return [opt], [sch]