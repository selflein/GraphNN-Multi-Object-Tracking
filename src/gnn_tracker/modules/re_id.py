from torch import nn
from torchreid.models.osnet import osnet_x0_5


class ReID(nn.Module):
    def __init__(self, out_feats, pretrained=True):
        super(ReID, self).__init__()
        self.osnet = osnet_x0_5(pretrained=pretrained)
        self.down = nn.Sequential(nn.Linear(512, 256),
                                  nn.LeakyReLU(),
                                  nn.Linear(256, out_feats))

    def forward(self, inp):
        out = self.osnet(inp)
        out = self.down(out)
        return out
