import torch
from torch import Tensor, nn

"""
Stand: Standardization

Net: Network

See Section 5.1

See formula (12) --> formula (13)

See Figure 1
"""


class StandNet(nn.Module):
    def __init__(self, p=2):
        """

        :param p: the order of norm
        """
        super(StandNet, self).__init__()
        self.p = p

    def forward(self, X: Tensor) -> Tensor:
        """

        :param X: shape=[n_sample, n_feature]
        :return: shape=[n_sample, n_feature] ||X[i,:]||_2 = 1
        """
        # print('StandNet.forward() ' + str(X.shape))
        norm = torch.norm(input=X, dim=1, p=self.p)
        I0 = norm == 0
        d = 1.0 / norm
        d[I0] = 0
        return torch.matmul(input=torch.diag_embed(input=d), other=X)
