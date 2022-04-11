import paddle
import paddle.nn as nn
from paddle.vision.models.resnet import resnet18, resnet34, resnet50, resnet101

class ResNet_MS3(nn.Layer):

    def __init__(self, depth=18, pretrained=False):
        super(ResNet_MS3, self).__init__()
        if depth == 18:
            net = resnet18(pretrained=pretrained)
        elif depth == 34:
            net = resnet34(pretrained=pretrained)
        elif depth == 50:
            net = resnet50(pretrained=pretrained)
        elif depth == 101:
            net = resnet101(pretrained=pretrained)
        # ignore the last block and fc
        self.model = paddle.nn.Sequential(*(list(net.children())[:-2]))

    def forward(self, x):
        res = []
        for name, module in self.model._sub_layers.items():
            x = module(x)
            if name in ['4', '5', '6']:
                res.append(x)
        return res


class ResNet_MS3_EXPORT(nn.Layer):

    def __init__(self, student, teacher):
        super(ResNet_MS3_EXPORT, self).__init__()
        self.student = student
        self.teacher = teacher

    def forward(self, x):
        result = []
        result.append(self.student(x))
        result.append(self.teacher(x))
        return result