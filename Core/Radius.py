import torch
from torch import nn, Tensor, optim
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
import numpy as np
from torch.optim import Optimizer

"""
(1) Ball: See Definition A.6 in Appendix A.2

(2) Radius <==> Min Ball: See Definition A.7 in Appendix A.2

(3) Min Ball <==> A Convex Optimization Problem: See Theorem 3.2 and Proof in Appendix A.2

(4) See Algorithm 1 Solving $g_1(\varphi)$ in Formula (10) in Appendix B.1

(5) See Section 3.3.2
"""


class Radius(nn.Module):
    __constants__ = ['sample_num', 'X', 'K']
    weight: Tensor

    def __init__(self, sample_num, X: Tensor = None, K: Tensor = None):
        """

        :param sample_num: integer
        :param X: sample matrix, shape=[sample_num, feature_num]
        :param K: Kernel matrix, shape=[sample_num, sample_num], K[i,j]=k(x_i)^T*k(x_j)
        """
        super(Radius, self).__init__()
        self.sample_num = sample_num
        self.X = None
        if X is not None:
            self.X = X.T  # Line 5

        self.K = K  # Line 3
        self.eps = 1e-10 / self.sample_num
        self.weight = Parameter(torch.Tensor(sample_num, 1))
        with torch.no_grad():
            self.__Mean_Init()

    def __Mean_Init(self):
        """
        Algorithm 1 Line 1
        :return:
        """
        init.constant_(self.weight, 1.0)

    def __forward_theta(self):
        theta = F.softmax(input=self.weight, dim=0)
        # theta[theta < self.eps] = 0.0
        return theta

    def __forward_X(self):
        """
        Line 10, 18
        :return:
        """
        theta = self.__forward_theta()
        c = torch.matmul(input=self.X, other=theta)  # shape=[feature_num, 1]
        dis = self.X - c  # shape=[feature_num, point_num]
        dis = dis * dis
        dis = torch.sum(input=dis, dim=0)  # shape=[1, point_num]
        return torch.relu(torch.max(dis))

    def __forward_K(self):
        """
        Line 8, 16
        :return:
        """
        theta = self.__forward_theta()
        K_theta = torch.matmul(input=self.K, other=theta)  # shape=[sample_num, 1]
        theta_K_theta = torch.matmul(input=theta.T, other=K_theta)  # shape=[1, 1]
        D_k = torch.diag(input=self.K).view([-1])  # shape=[sample_num]
        K_theta = K_theta.view([-1])  # shape=[sample_num]
        theta_K_theta = theta_K_theta.view(-1)  # shape=1
        dis = D_k - 2 * K_theta + theta_K_theta  # shape=[sample_num]
        return torch.relu(torch.max(dis))

    def __radius2(self):
        """
        Lines 7-11, 15-19
        :return:
        """
        if self.X is not None:
            return self.__forward_X()
        else:
            return self.__forward_K()

    def solving(self, max_ite_num=10000, min_ite_gap=1e-10, lr=1e-3, decay_how_often=None, decay_factor=0.5):
        """

        :param max_ite_num:
        :param min_ite_gap:
        :param lr:
        :param decay_how_often:
        :param decay_factor:
        :return:
        """
        if decay_how_often is None:
            decay_how_often = max_ite_num
        max_ite_num = torch.tensor(data=max_ite_num, device=self.weight.device)
        ite_num = torch.tensor(data=0, device=self.weight.device)

        min_ite_gap = torch.tensor(data=min_ite_gap, device=self.weight.device)
        old_r2 = torch.tensor(data=0, device=self.weight.device)
        new_r2 = torch.tensor(data=np.inf, device=self.weight.device)
        min_r2 = torch.tensor(data=np.inf, device=self.weight.device)
        best_theta = None

        optimizer = optim.Adam(params=self.parameters(), lr=lr)
        while ite_num < max_ite_num and torch.abs(new_r2 - old_r2) > min_ite_gap:  # Line 12
            old_r2 = new_r2

            optimizer.zero_grad()
            new_r2 = self.__radius2()
            new_r2.backward()
            optimizer.step()  # Line 14

            new_r2 = new_r2.detach().clone()
            if new_r2 < min_r2:
                best_theta = self.__forward_theta()
                best_theta.detach().clone()
                min_r2 = new_r2

            ite_num += 1

            if ite_num % decay_how_often == 0:
                Radius.Decay_lr(optimizer=optimizer, decay_factor=decay_factor)

            # print('ite= ' + str(ite_num) + ' r^2= ' + str(new_r2))

        return {'R2': min_r2.detach().clone().cpu().numpy(),
                'theta': best_theta}

    @staticmethod
    def Decay_lr(optimizer: Optimizer, decay_factor=0.5):
        for para_group in optimizer.param_groups:
            para_group['lr'] = para_group['lr'] * decay_factor
