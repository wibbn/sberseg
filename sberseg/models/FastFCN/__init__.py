import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from sberseg.models.FastFCN.resnet import Bottleneck, ResNet
from sberseg.models.FastFCN.parts import JPU, Head
from sberseg.utils.metrics import SegmentationMetric

from sberseg.utils.data import get_blended_output, save_image, wb_image

class FastFCN(pl.LightningModule):
    def __init__(self, num_classes=8, learning_rate=1e-3, bn_momentum=0.01, out_dir='data/output'):
        super(FastFCN, self).__init__()

        self.learning_rate = learning_rate
        self.out_dir = out_dir
        self.metrics = SegmentationMetric(num_classes)

        self.resnet50 = ResNet(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            dilation=[1, 1, 1, 2],
            bn_momentum=bn_momentum,
            is_fpn=True
        )

        self.jpu = JPU([512, 1024, 2048], width=512, norm_layer=nn.BatchNorm2d)
        self.head = Head(num_classes, norm_layer=nn.BatchNorm2d)

        self.save_hyperparameters('num_classes', 'learning_rate')

    def forward(self, input):
        blocks = self.resnet50(input)
        f1, f2, f3, f4 = self.jpu(blocks)
        pred = self.head([f1, f2, f3, f4])

        output = F.interpolate(pred, size=input.size()[2:4], mode='bilinear', align_corners=True)
        return output

    def training_step(self, batch, batch_nb) :
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

    def validation_step(self, batch, batch_nb) :
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
        # sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=2)
        return [opt], [sch]


if __name__ == '__main__':
    in_batch, inchannel, in_h, in_w = 4, 3, 128, 128
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = FastFCN(num_classes=8)
    # out = net(x)
    # print(out.shape)
    # print(out[0])
    
    a = [i for i in net.parameters()]
    b = [i for i in net.resnet50.parameters()]
    c = [i for i in net.jpu.parameters()]
    d = [i for i in net.head.parameters()]

    print(len(a), len(b), len(c), len(d), len(b) + len(c) + len(d))
