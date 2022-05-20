import torch
from torch import nn
from models.non_local import NONLocalBlock3D


class SB1(nn.Module):

    def __init__(self):
        super(SB1, self).__init__()

        self.maxpool = nn.MaxPool3d(kernel_size=(4, 14, 14))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(in_features=1024, out_features=2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.avgpool(x).view(batch_size, -1)
        x = self.fc(x)

        return x


class SB3(nn.Module):

    def __init__(self):
        super(SB3, self).__init__()

        self.maxpool = nn.MaxPool3d(kernel_size=(4, 14, 14))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=2)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.avgpool(x).view(batch_size, -1)
        x = self.fc(x)

        return x


class SB_nl(nn.Module):

    def __init__(self):
        super(SB_nl, self).__init__()

        self.nl = NONLocalBlock3D(in_channels=1024)
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 14, 14))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(in_features=1024, out_features=2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.nl(x)
        x = self.avgpool(x).view(batch_size, -1)
        x = self.fc(x)

        return x


class SBF_fc(nn.Module):

    def __init__(self):
        super(SBF_fc, self).__init__()

        self.maxpool = nn.MaxPool3d(kernel_size=(4, 14, 14))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=1024, out_features=2)

    def forward(self, x_body, x_face):
        batch_size = x_body.size(0)
        x_body = self.avgpool(x_body).view(batch_size, -1)
        x_body = self.fc1(x_body)
        x = torch.cat((x_body, x_face), dim=1)
        x = self.fc2(x)

        return x


class SBF_nlfc(nn.Module):

    def __init__(self):
        super(SBF_nlfc, self).__init__()

        self.nl = NONLocalBlock3D(in_channels=1024)
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 14, 14))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=1024, out_features=2)

    def forward(self, x_body, x_face):
        batch_size = x_body.size(0)
        x_body = self.nl(x_body)
        x_body = self.avgpool(x_body).view(batch_size, -1)
        x_body = self.fc1(x_body)
        x = torch.cat((x_body, x_face), dim=1)
        x = self.fc2(x)

        return x


class SBF_nlfc_IBF_nlfc(nn.Module):

    def __init__(self):
        super(SBF_nlfc_IBF_nlfc, self).__init__()

        self.nl1 = NONLocalBlock3D(in_channels=1024)
        self.nl2 = NONLocalBlock3D(in_channels=1024)
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 14, 14))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=2048, out_features=2)

    def forward(self, x_self_body, x_self_face, x_interact_body, x_interact_face):
        batch_size = x_self_body.size(0)

        x_self_body = self.nl1(x_self_body)
        x_self_body = self.avgpool(x_self_body).view(batch_size, -1)
        x_self_body = self.fc1(x_self_body)
        x_self = torch.cat((x_self_body, x_self_face), dim=1)

        x_interact_body = self.nl2(x_interact_body)
        x_interact_body = self.avgpool(x_interact_body).view(batch_size, -1)
        x_interact_body = self.fc2(x_interact_body)
        x_interact = torch.cat((x_interact_body, x_interact_face), dim=1)

        x = torch.cat((x_self, x_interact), dim=1)
        x = self.fc3(x)

        return x


class SBF_nlfc_IBF_nlfc_d(nn.Module):

    def __init__(self):
        super(SBF_nlfc_IBF_nlfc_d, self).__init__()

        self.nl1 = NONLocalBlock3D(in_channels=1024)
        self.nl2 = NONLocalBlock3D(in_channels=1024)
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 14, 14))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=2048, out_features=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x_self_body, x_self_face, x_interact_body, x_interact_face):
        batch_size = x_self_body.size(0)

        x_self_body = self.nl1(x_self_body)
        x_self_body = self.avgpool(x_self_body).view(batch_size, -1)
        x_self_body = self.fc1(x_self_body)
        x_self_body = self.dropout(x_self_body)
        x_self = torch.cat((x_self_body, x_self_face), dim=1)

        x_interact_body = self.nl2(x_interact_body)
        x_interact_body = self.avgpool(x_interact_body).view(batch_size, -1)
        x_interact_body = self.fc2(x_interact_body)
        x_interact_body = self.dropout(x_interact_body)
        x_interact = torch.cat((x_interact_body, x_interact_face), dim=1)

        x = torch.cat((x_self, x_interact), dim=1)
        x = self.fc3(x)
        x = self.dropout(x)

        return x


class SBF_nlfc_IBF_nlfc_d_reg(nn.Module):

    def __init__(self):
        super(SBF_nlfc_IBF_nlfc_d_reg, self).__init__()

        self.nl1 = NONLocalBlock3D(in_channels=1024)
        self.nl2 = NONLocalBlock3D(in_channels=1024)
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 14, 14))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=2048, out_features=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x_self_body, x_self_face, x_interact_body, x_interact_face):
        batch_size = x_self_body.size(0)

        x_self_body = self.nl1(x_self_body)
        x_self_body = self.avgpool(x_self_body).view(batch_size, -1)
        x_self_body = self.fc1(x_self_body)
        x_self_body = self.dropout(x_self_body)
        x_self = torch.cat((x_self_body, x_self_face), dim=1)

        x_interact_body = self.nl2(x_interact_body)
        x_interact_body = self.avgpool(x_interact_body).view(batch_size, -1)
        x_interact_body = self.fc2(x_interact_body)
        x_interact_body = self.dropout(x_interact_body)
        x_interact = torch.cat((x_interact_body, x_interact_face), dim=1)

        x = torch.cat((x_self, x_interact), dim=1)
        x = self.fc3(x)

        return x
