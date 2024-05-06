import torch
import torch.nn as nn
import utils
import numpy as np
from torch.utils.tensorboard import SummaryWriter

##define the loss
class UM_loss(nn.Module):
    #initial loss parameters
    def __init__(self, alpha, beta, margin):
        super(UM_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.ce_criterion = nn.BCELoss()##Binary cross-entropy loss function
##forward transfoming,input is i3d output:action score,background score,action feature,bankground feature,label
    def forward(self, score_act, score_bkg, feat_act, feat_bkg, label):
        # print(features)
        # print(features.shape)
        loss = {}

        label = label / torch.sum(label, dim=1, keepdim=True)
##caculate classify loss，using Binary cross-entropy loss function
        loss_cls = self.ce_criterion(score_act, label)
##caculate boundary loss,label-bkgis 1,which means all the sampels is background
        label_bkg = torch.ones_like(label).cuda()
        label_bkg /= torch.sum(label_bkg, dim=1, keepdim=True)
        loss_be = self.ce_criterion(score_bkg, label_bkg)
##caculate act,bkg loss,Euclidean distance.
        loss_act = self.margin - torch.norm(torch.mean(feat_act, dim=1), p=2, dim=1)
        loss_act[loss_act < 0] = 0
        loss_bkg = torch.norm(torch.mean(feat_bkg, dim=1), p=2, dim=1)
##unsupervised loss,The average of the sum of squared losses
        loss_um = torch.mean((loss_act + loss_bkg) ** 2)
## sum,alpha,beta adjust loss
        loss_total = loss_cls + self.alpha * loss_um + self.beta * loss_be

        loss["loss_cls"] = loss_cls
        loss["loss_be"] = loss_be
        loss["loss_um"] = loss_um
        loss["loss_total"] = loss_total

        return loss_total, loss

##train cnn,net(cnn model),iteration(provide training data),update para,loss criterion,log,step time
def train(net, loader_iter, optimizer, criterion, logger, step):
    net.train()
    ##dataiterator loading got next batch data and label
    _data, _label, _, _, _ = next(loader_iter)
##move to the cpu to compute
    _data = _data.cuda()
    _label = _label.cuda()
## the gradients initial
    optimizer.zero_grad()

    score_act, score_bkg, feat_act, feat_bkg, _, _ = net(_data)##out
# comput total loss,dictionary
    cost, loss = criterion(score_act, score_bkg, feat_act, feat_bkg, _label)
##update
    cost.backward()
    optimizer.step()
##traverse,track the change
    for key in loss.keys():
        logger.log_value(key, loss[key].cpu().item(), step)


    # create TensorBoardX SummaryWriter，store path make sure
    writer = SummaryWriter(log_dir='.log/events')

    # writer.add_scalar()record data
    # comuse loss contain loss things
    for key in loss.keys():
        writer.add_scalar(key, loss[key].cpu().item(), step)

    # record weight, gradients in tensorboard
    ##histogram Stacked's frequency distribution
    for name, param in net.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), step)
        if param.grad is not None:
            writer.add_histogram(name + '_grad', param.grad.clone().cpu().data.numpy(), step)

    writer.close()
