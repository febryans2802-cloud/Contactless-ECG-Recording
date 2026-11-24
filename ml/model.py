"""Small 1D U-Net: 8 s of 30 Hz PPG -> 8 s of 120 Hz ECG.

Only Conv1d (odd kernel, 'same' padding), LeakyReLU, MaxPool(2),
nearest Upsample(2) and channel concat are used, so the network can be
re-implemented in a few lines of plain JavaScript (src/ecgModel.js).
"""
import torch
import torch.nn as nn

K = 7


def block(cin, cout):
    return nn.Sequential(
        nn.Conv1d(cin, cout, K, padding=K // 2), nn.LeakyReLU(0.1),
        nn.Conv1d(cout, cout, K, padding=K // 2), nn.LeakyReLU(0.1))


class PPG2ECG(nn.Module):
    def __init__(self, c=(24, 48, 96)):
        super().__init__()
        c1, c2, c3 = c
        self.e1, self.e2, self.e3 = block(1, c1), block(c1, c2), block(c2, c3)
        self.mid = block(c3, c3)
        self.d3, self.d2, self.d1 = block(2 * c3, c2), block(2 * c2, c1), block(2 * c1, c1)
        self.u1 = block(c1, c1)   # 240 -> 480
        self.u2 = block(c1, 16)   # 480 -> 960
        self.head = nn.Conv1d(16, 1, K, padding=K // 2)
        self.pool = nn.MaxPool1d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):            # x: [B,1,240]
        a = self.e1(x)               # 240
        b = self.e2(self.pool(a))    # 120
        c = self.e3(self.pool(b))    # 60
        m = self.mid(self.pool(c))   # 30
        y = self.d3(torch.cat([self.up(m), c], 1))
        y = self.d2(torch.cat([self.up(y), b], 1))
        y = self.d1(torch.cat([self.up(y), a], 1))
        y = self.u1(self.up(y))
        y = self.u2(self.up(y))
        return self.head(y)          # [B,1,960]
