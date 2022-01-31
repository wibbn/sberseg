import torch
import torchvision
import pytorch_lightning as pl
from torch.nn import functional as F

class FCN(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super(FCN, self).__init__()
        self.learning_rate = learning_rate
        self.resnet50 = torchvision.models.segmentation.fcn_resnet50(
            pretrained = False, 
            progress = True, 
            num_classes = num_classes
        )
        
    def forward(self, x):
        return self.resnet50(x)['out']
    
    def training_step(self, batch, batch_nb) :
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)
        loss_val = F.cross_entropy(out, mask, ignore_index = 250)
        return {'loss' : loss_val}
    
    def training_step_end(self, outs):
        self.log("train/loss", outs['loss'])

    def validation_step(self, batch, batch_nb) :
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)

        loss_val = F.cross_entropy(out, mask, ignore_index = 250)
        return {'loss' : loss_val}
    
    def validation_step_end(self, outs):
        self.log("val/loss", outs['loss'])
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        # sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=2)
        return [opt], [sch]

if __name__ == '__main__':
    in_batch, inchannel, in_h, in_w = 4, 3, 128, 128
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = FCN(num_classes=8)
    out = net(x)
    print(out.shape)
    print(out)