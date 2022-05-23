import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------
# DMUE
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_R18(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_R18, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck_R18, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)    # add missed relu
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class DMUE(nn.Module):
    def __init__(self, bnneck=True):
        super(DMUE, self).__init__()
        last_stride = 2
        model_name = 'resnet18'
        self.num_classes = 8
        self.num_branches = 9

        # base model
        self.in_planes = 512
        self.base = ResNet18(last_stride=last_stride, block=BasicBlock, layers=[2, 2, 2, 2])

        # pooling after base
        self.gap = nn.AdaptiveAvgPool2d(1)

        # loss
        self.classifiers = []
        self.classifiers.append(nn.Linear(self.in_planes, self.num_classes, bias=False))
        self.classifiers = nn.ModuleList(self.classifiers)
        
        self.neck = bnneck
        if bnneck:
            self.bottleneck = nn.BatchNorm1d(self.in_planes)


    def forward(self, x):
        # if batch_size = 16
        # x: [16, 3, 224, 224]
        x_final = self.base(x)                                          # [16, 512, 7, 7]
        x_final = self.gap(x_final).squeeze(2).squeeze(2)               # [16, 512]
        # x_final = self.bottleneck(x_final) if self.neck else x_final  # [16, 512]
        # x_final = self.classifiers[0](x_final)                        # [16, 8]
                              
        return x_final

    def load_param(self):
        pretrained = 'pretrained/dmue_r18_affectnet.pth'
        param_dict = torch.load(pretrained, map_location=lambda storage,loc: storage.cpu())

        for i in param_dict:
            if 'fc.' in i: continue
            self.state_dict()[i].copy_(param_dict[i])


# ------------------------------------------------------------------------------
# Non-local block
class NonLocalBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 sub_sample=True,
                 bn_layer=True):

        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv3d
        max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
        bn = nn.BatchNorm3d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        x             : (b, c, t, h, w)
        return_nl_map : if True return z, nl_map, else only return z.
        """
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


# ------------------------------------------------------------------------------
# ResNet-50 3D
class FrozenBN(nn.Module):

    def __init__(self, num_channels, momentum=0.1, eps=1e-5):
        super(FrozenBN, self).__init__()
        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps
        self.params_set = False

    def set_params(self, scale, bias, running_mean, running_var):
        self.register_buffer('scale', scale)
        self.register_buffer('bias', bias)
        self.register_buffer('running_mean', running_mean)
        self.register_buffer('running_var', running_var)
        self.params_set = True

    def forward(self, x):
        assert self.params_set, 'model.set_params() must be called before forward pass'
        return torch.batch_norm(x, self.scale, self.bias, self.running_mean,
                                self.running_var, False, self.momentum,
                                self.eps, torch.backends.cudnn.enabled)

    def __repr__(self):
        return 'FrozenBN(%d)' % self.num_channels


def freeze_bn(m, name):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.BatchNorm3d:
            frozen_bn = FrozenBN(target_attr.num_features,
                                 target_attr.momentum, target_attr.eps)
            frozen_bn.set_params(target_attr.weight.data,
                                 target_attr.bias.data,
                                 target_attr.running_mean,
                                 target_attr.running_var)
            setattr(m, attr_str, frozen_bn)
    for n, ch in m.named_children():
        freeze_bn(ch, n)


class Bottleneck_R3D(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride,
                 downsample,
                 temp_conv,
                 temp_stride,
                 use_nl=False):
        super(Bottleneck_R3D, self).__init__()
        self.conv1 = nn.Conv3d(inplanes,
                               planes,
                               kernel_size=(1 + temp_conv * 2, 1, 1),
                               stride=(temp_stride, 1, 1),
                               padding=(temp_conv, 0, 0),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes,
                               planes,
                               kernel_size=(1, 3, 3),
                               stride=(1, stride, stride),
                               padding=(0, 1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes,
                               planes * 4,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        outplanes = planes * 4

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):

    def __init__(self,
                 block=Bottleneck_R3D,
                 layers=[3, 4, 6, 3],
                 num_classes=400,
                 use_nl=False):
        self.inplanes = 64
        super(ResNet3D, self).__init__()
        self.conv1 = nn.Conv3d(3,
                               64,
                               kernel_size=(5, 7, 7),
                               stride=(2, 2, 2),
                               padding=(2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 3, 3),
                                     stride=(2, 2, 2),
                                     padding=(0, 0, 0))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1),
                                     stride=(2, 1, 1),
                                     padding=(0, 0, 0))

        nonlocal_mod = 2 if use_nl else 1000
        self.layer1 = self._make_layer(block,
                                       64,
                                       layers[0],
                                       stride=1,
                                       temp_conv=[1, 1, 1],
                                       temp_stride=[1, 1, 1])
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       temp_conv=[1, 0, 1, 0],
                                       temp_stride=[1, 1, 1, 1],
                                       nonlocal_mod=nonlocal_mod)
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       temp_conv=[1, 0, 1, 0, 1, 0],
                                       temp_stride=[1, 1, 1, 1, 1, 1],
                                       nonlocal_mod=nonlocal_mod)
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       temp_conv=[0, 1, 0],
                                       temp_stride=[1, 1, 1])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.drop = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride,
                    temp_conv,
                    temp_stride,
                    nonlocal_mod=1000):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or temp_stride[
                0] != 1:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=(1, 1, 1),
                          stride=(temp_stride[0], stride, stride),
                          padding=(0, 0, 0),
                          bias=False),
                nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, temp_conv[0],
                  temp_stride[0], False))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, 1, None, temp_conv[i],
                      temp_stride[i], i % nonlocal_mod == nonlocal_mod - 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # print(f'conv1 output size {x.size()}')     # [b,64,16,112,112]
        x = self.bn1(x)
        # print(f'bn1 output size {x.size()}')       # [b,64,16,112,112]
        x = self.relu(x)
        x = self.maxpool1(x)
        # print(f'maxpool1 output size {x.size()}')  # [b,64,8,55,55]

        x = self.layer1(x)
        # print(f'layer1 output size {x.size()}')    # [b,256,8,55,55]
        x = self.maxpool2(x)
        # print(f'maxpool2 output size {x.size()}')  # [b,256,4,55,55]
        x = self.layer2(x)
        # print(f'layer2 output size {x.size()}')    # [b,512,4,28,28]
        x = self.layer3(x)
        # print(f'layer3 output size {x.size()}')    # [b,1024,4,14,14]
        # x = self.layer4(x)
        # print(f'layer4 output size {x.size()}')    # [b,2048,4,7,7]

        # x = self.avgpool(x)
        # print(f'avgpool output size {x.size()}')   # [b,2048,1,1,1]

        return x


class MyModel(nn.Module):

    def __init__(self, ResNet3D, DMUE):
        super(MyModel, self).__init__()

        self.SBody = ResNet3D
        self.SFace = DMUE
        self.IBody = ResNet3D
        self.IFace = DMUE

        self.nl1 = NonLocalBlock(in_channels=1024)
        self.nl2 = NonLocalBlock(in_channels=1024)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=2048, out_features=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, s_body, s_face, i_body, i_face):
        batch_size = s_body.size(0)

        s_body = self.SBody(s_body)
        s_body = self.nl1(s_body)
        s_body = self.avgpool(s_body).view(batch_size, -1)
        s_body = self.fc1(s_body)
        s_body = self.dropout(s_body)
        s_face = self.SFace(s_face)
        s_face = s_face.reshape(-1, 4, 512)
        s_face = torch.mean(s_face, 1)
        s_out = torch.cat((s_body, s_face), dim=1)

        i_body = self.IBody(i_body)
        i_body = self.nl2(i_body)
        i_body = self.avgpool(i_body).view(batch_size, -1)
        i_body = self.fc2(i_body)
        i_body = self.dropout(i_body)
        i_face = self.IFace(i_face)
        i_face = i_face.reshape(-1, 4, 512)
        i_face = torch.mean(i_face, 1)
        i_out = torch.cat((i_body, i_face), dim=1)

        x = torch.cat((s_out, i_out), dim=1)
        x = self.fc3(x)

        return x


class MyModelS(nn.Module):

    def __init__(self, ResNet3D, DMUE):
        super(MyModelS, self).__init__()

        self.Body = ResNet3D
        self.Face = DMUE

        self.nl = NonLocalBlock(in_channels=1024)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=1024, out_features=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, body, face):
        batch_size = body.size(0)

        body = self.Body(body)
        body = self.nl(body)
        body = self.avgpool(body).view(batch_size, -1)
        body = self.fc1(body)
        body = self.dropout(body)
        face = self.Face(face)
        face = face.reshape(-1, 4, 512)
        face = torch.mean(face, 1)
        out = torch.cat((body, face), dim=1)

        out = self.fc2(out)

        return out


class MyModelSBody(nn.Module):

    def __init__(self, ResNet3D):
        super(MyModelSBody, self).__init__()

        self.Body = ResNet3D
        self.nl = NonLocalBlock(in_channels=1024)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, body):
        batch_size = body.size(0)

        body = self.Body(body)
        body = self.nl(body)
        body = self.avgpool(body).view(batch_size, -1)
        body = self.fc1(body)
        body = self.dropout(body)
        out = self.fc2(body)

        return out


class MyModelSFace(nn.Module):

    def __init__(self, DMUE):
        super(MyModelSFace, self).__init__()

        self.Face = DMUE
        self.fc = nn.Linear(in_features=512, out_features=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, face):
        face = self.Face(face)
        face = face.reshape(-1, 4, 512)
        face = torch.mean(face, 1)
        out = self.fc(face)

        return out


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # sample data
    # r3d  : [b, 3, 32, 224, 224]
    # dmue : [b, 3, 224, 224]
    batch_size = 4
    s_body = torch.randn(batch_size, 3, 32, 224, 224)
    s_face = torch.randn(batch_size, 3,  4, 224, 224)
    i_body = torch.randn(batch_size, 3, 32, 224, 224)
    i_face = torch.randn(batch_size, 3,  4, 224, 224)
    print(s_body.size(), s_face.size(), i_body.size(), i_face.size())

    s_face = s_face.permute(0, 2, 1, 3, 4).reshape(-1, 3, 224, 224)
    i_face = i_face.permute(0, 2, 1, 3, 4).reshape(-1, 3, 224, 224)

    s_body = s_body.to(device='cuda')
    s_face = s_face.to(device='cuda')
    i_body = i_body.to(device='cuda')
    i_face = i_face.to(device='cuda')
    print(s_body.size(), s_face.size(), i_body.size(), i_face.size())

    # --------------------------------------------------------------------------
    r3d = ResNet3D(num_classes=400)
    r3d.load_state_dict(torch.load('pretrained/i3d_r50_kinetics.pth'))
    # r3d.cuda()
    # freeze_bn(r3d, "net")  # Used for finetune. For validation, .eval() works.
    # out = r3d(s_body)
    # print(out.size())

    dmue = DMUE(bnneck=True)
    dmue.load_param()
    # dmue = dmue.cuda()
    # out = model(s_face).reshape(-1, 4, 512)
    # print(out.size())

    net_1 = MyModel(r3d, dmue)
    net_1.cuda()
    out = net_1(s_body, s_face, i_body, i_face)
    print(out.size())

    net_2 = MyModelS(r3d, dmue)
    net_2.cuda()
    out = net_2(s_body, s_face)
    print(out.size())
