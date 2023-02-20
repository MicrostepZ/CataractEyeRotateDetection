from nets.utils.CondConv import *
from nets.utils.CoordAttention import *

class AttConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AttConvBlock,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.ca = CoordAtt(in_channel, in_channel)
        self.c3 = nn.Sequential(
            CondConv(in_channel, in_channel, kernel_size=3, stride=1, padding=1, grounps=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.c3(x)
        x2 = self.ca(x1)
        x3 = self.c1(x2)
        return x+x3

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            AttConvBlock(32,32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downroute1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),)

        self.downroute2 = nn.Sequential(
            AttConvBlock(64, 64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=1, dilation=2),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True),
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=2, dilation=4),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True),
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=4, dilation=8),
        )
        self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_1.data.fill_(0.5)
        self.fuse_weight_2.data.fill_(0.5)

    def forward(self, x):
        x1 = self.head(x)
        x2 = self.pool(x1)
        x2 = self.downroute2(x2)

        des_size = x.shape[2:]

        x2 = F.interpolate(x2, des_size, mode='bilinear')

        x3 = self.downroute1(x1)

        out = x2 * self.fuse_weight_1 + x3 * self.fuse_weight_2

        return out

class BaseNet(nn.Module):
    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:,1:2]

    def normalize(self, x, ureliability, urepeatability):
        return dict(descriptors = F.normalize(x, p=2, dim=1),
                    repeatability = self.softmax( urepeatability ),
                    reliability = self.softmax( ureliability ))

    def forward_one(self, x):
        raise NotImplementedError()

    def forward(self, imgs, **kw):
        res = [self.forward_one(img) for img in imgs]
        # merge all dictionaries into one
        res = {k:[r[k] for r in res if k in r] for k in {k for r in res for k in r}}
        return dict(res, imgs=imgs, **kw)

class ournet(BaseNet):
    def __init__(self, **kw):
        BaseNet.__init__(self, **kw)
        self.out_dim = 128

        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)

        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1)

        self.backbone = Backbone()

    def forward_one(self, x):
        x = self.backbone(x)
        # compute the confidence maps
        ureliability = self.clf(x ** 2)
        urepeatability = self.sal(x ** 2)
        return self.normalize(x, ureliability, urepeatability)













