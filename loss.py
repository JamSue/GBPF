from torch import nn
import torch
from torch.nn import functional as F
from config import args
# 三元度量损失
class TripleLoss(nn.Module):
    def __init__(self, num_class, size_average=True):
        super(TripleLoss, self).__init__()
        self.num_class = num_class
        self.size_average = size_average
        
    def forward(self, classes, labels,dis_same, dis_dif):
        """
        classes: 是模型预测得到的关于标签的结果
        labels: 是样本的真实标签
        """
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        # loss = self.cross_entropy_loss(classes, labels) +  dis_same*0.01 - dis_dif*0.01
        entr_loss = self.cross_entropy_loss(classes, labels)
        # print("entr_loss:",entr_loss)
        # print("dis_same:",dis_same)
        # print("dis_dif:",dis_dif)
        loss = self.cross_entropy_loss(classes, labels)  +  args.a*dis_same - args.b*dis_dif # AGNEWS
       
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

# 对比损失
class ContrastiveLioss(nn.Module):
    def __init__(self, num_class, size_average=True):
        super(TripleLoss, self).__init__()
        self.num_class = num_class
        self.size_average = size_average
        
    def forward(self, classes, labels,dis_same, dis_dif):
        """
        classes: 是模型预测得到的关于标签的结果
        labels: 是样本的真实标签
        """
        t = 0.5  # 温度系数
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        ceLoss = self.cross_entropy_loss(classes, labels) 
        batch_size = labels.shape(0)
        loss = ceLoss - 1/batch_size
       
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
            