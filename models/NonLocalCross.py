import torch
from torch import nn
from torch.nn import functional as F


class _BiNonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_BiNonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g_a = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.g_b = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0)
        if bn_layer:
            self.W_a = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W_a[1].weight, 0)
            nn.init.constant_(self.W_a[1].bias, 0)
            
            self.W_b = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W_b[1].weight, 0)
            nn.init.constant_(self.W_b[1].bias, 0)
        else:
            self.W_a = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W_a.weight, 0)
            nn.init.constant_(self.W_a.bias, 0)
            
            self.W_b = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W_b.weight, 0)
            nn.init.constant_(self.W_b.bias, 0)

        # if sub_sample:
        # self.g_a = nn.Sequential(self.g_a, max_pool_layer)
        # self.g_b = nn.Sequential(self.g_b, max_pool_layer)
        # self.phi = nn.Sequential(self.phi, max_pool_layer)
        # self.theta = nn.Sequential(self.theta, max_pool_layer)
        

    def forward(self, a, b):
        '''
        :param a, b: (b, c, t, h, w)
        :return:
        '''
        # v = k
        # import pdb; pdb.set_trace()
        batch_size = a.size(0)

        v_a = self.g_a(a).view(batch_size, self.inter_channels, -1) # value
        v_a = v_a.permute(0, 2, 1)
        
        
        v_b = self.g_b(b).view(batch_size, self.inter_channels, -1) # value
        v_b = v_b.permute(0, 2, 1)

        q_a = self.theta(a).view(batch_size, self.in_channels, -1)
        q_a = q_a.permute(0, 2, 1)

        # if self.sub_sample:
        k_b = self.phi(b).view(batch_size, self.in_channels, -1)
        # else:
        #     k_b = b.view(batch_size, self.in_channels, -1)

        f = torch.matmul(q_a, k_b) # q, k
        f_t = f.clone().transpose(1, 2)
        f_div_C = F.softmax(f, dim=-1)
        f_div_C_T = F.softmax(f_t, dim=-1)

        y = torch.matmul(f_div_C, v_a)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *a.size()[2:])
        W_y = self.W_a(y)
        z = W_y + a
        
        
        # z_inverse
        y_inv = torch.matmul(f_div_C_T, v_b)
        y_inv = y_inv.permute(0, 2, 1).contiguous()
        y_inv = y_inv.view(batch_size, self.inter_channels, *b.size()[2:])
        W_y_inv = self.W_b(y_inv)
        z_inv = W_y_inv + b
        
        return z, z_inv


class BiNonLocalBlock1D(_BiNonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(BiNonLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class BiNonLocalBlock2D(_BiNonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(BiNonLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class BiNonLocalBlock3D(_BiNonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(BiNonLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


if __name__ == '__main__':
    import torch

    for (sub_sample, bn_layer) in [(True, True), (False, False), (True, False), (False, True)]:
        # img = torch.zeros(2, 3, 20)
        # net = BiNonLocalBlock1D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        # out1, out2 = net(img, img)
        # print(out1.size())
        # print(out2.size())
        
        # import pdb; pdb.set_trace()
        img = torch.zeros(2, 3, 20, 20)
        net = BiNonLocalBlock2D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out1, out2 = net(img, img)
        print(out1.size())
        print(out2.size())

        img = torch.randn(2, 3, 8, 20, 20)
        net = BiNonLocalBlock3D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out1, out2 = net(img, img)
        print(out1.size())
        print(out2.size())
