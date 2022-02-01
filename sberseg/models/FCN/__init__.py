import torch
import torchvision
import pytorch_lightning as pl
from torch.nn import functional as F

from sberseg.utils.metrics import SegmentationMetric
from sberseg.utils.data import get_blended_output, save_image, wb_image

class FCN(pl.LightningModule):
    def __init__(self, num_classes=8, learning_rate=1e-3, out_dir: str = 'data/output'):
        super(FCN, self).__init__()
        self.learning_rate = learning_rate
        self.out_dir=out_dir
        self.metrics = SegmentationMetric(num_classes)
        self.resnet50 = torchvision.models.segmentation.fcn_resnet50(
            pretrained = False, 
            progress = True, 
            num_classes = num_classes
        )

        self.save_hyperparameters('num_classes', 'learning_rate')
        
    def forward(self, x):
        return self.resnet50(x)['out']
    
    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)
        loss_val = F.cross_entropy(out, mask, ignore_index = 250)
        return {'loss' : loss_val}
    
    def training_step_end(self, outs):
        self.log("train/loss", outs['loss'])

    def log_images(self, imgs, masks, save=False):
        log_imgs = []
        for i, m in zip(imgs, masks):
            log_imgs.append(wb_image(i, m))
            if save: save_image(get_blended_output(i, m), self.out_dir)
            
        self.logger.experiment.log({'val/imgs': log_imgs})

    def validation_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)

        self.metrics.update(mask, out)
        self.log_images(img, out)

        loss_val = F.cross_entropy(out, mask, ignore_index = 250)
        return {'loss' : loss_val}
    
    def validation_step_end(self, outs):
        pixacc, miou = self.metrics.get()

        self.log("val/loss", outs['loss'])
        self.log("val/pixAcc", pixacc)
        self.log("val/mIoU", miou)

        self.metrics.reset()

    def test_step(self, img, batch_nb):
        img = img.float()
        out = self.forward(img)

        self.log_images(img, out, save=True)
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=2)
        return [opt], [sch]
