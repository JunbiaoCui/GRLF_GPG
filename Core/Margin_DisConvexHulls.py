import torch
from torch import nn, Tensor, optim
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
import numpy as np
from torch.optim import Optimizer

"""
(1) M, Margin: See Definition 2.2

(2) ch(A), Convex Hull: See Definition Definition A.1 in Appendix A.1

(3) dis(ch(A), ch(B)), Distance between Two Convex Hulls: See Definition A.2 in Appendix A.1

(4) M^2 >= 1/4 dis^2(ch(X_+), ch(X_-)), and 
    dis^2(ch(X_+), ch(X_-)) <==> A Convex Optimization Problem
    See Theorem 3.1 and Proof in in Appendix A.1

(5) See Algorithm 2 Solving $g_2(\varphi)$ in Formula(10) in Appendix B.1

(6) See Section 3.3.1
"""


class Margin_DisConvexHulls(nn.Module):
    __constants__ = ['A', 'B', 'K', 'A_sam_num', 'B_sam_num']
    A_weight: Tensor
    B_weight: Tensor

    def __init__(self, A_sam_num, B_sam_num, A: Tensor = None, B: Tensor = None, K: Tensor = None):
        """

        :param A_sam_num:
        :param B_sam_num:
        :param A: shape=[A_sam_num, feature_num]
        :param B: shape=[B_sam_num, feature_num]
        :param K: shape=[A_sam_num + B_sam_num, A_sam_num + B_sam_num]
               K=[K_AA,   K_AB;
                  K_AB^T, K_BB],
               K_AA[i,j]=k(a_i)^T * k(a_i)
               K_BB[i,j]=k(b_i)^T * k(b_i)
               K_AB[i,j]=k(a_i)^T * k(b_i)
               k: kernel map
        """
        super(Margin_DisConvexHulls, self).__init__()
        self.A_sam_num = A_sam_num
        self.B_sam_num = B_sam_num

        self.A = None
        self.B = None
        """ Algorithm 2 Line 5 """
        if A is not None:
            self.A = A.T
        if B is not None:
            self.B = B.T

        self.K = None
        if K is not None:
            """ Algorithm 2 Line 3 """
            n = self.A_sam_num + self.B_sam_num
            K[0:self.A_sam_num, self.A_sam_num:n] = - K[0:self.A_sam_num, self.A_sam_num:n]
            K[self.A_sam_num:n, 0:self.A_sam_num] = - K[self.A_sam_num:n, 0:self.A_sam_num]
            # print(K)
            self.K = K

        self.eps = 1e-10 / (self.A_sam_num + self.B_sam_num)
        self.A_weight = Parameter(torch.Tensor(self.A_sam_num, 1))
        self.B_weight = Parameter(torch.Tensor(self.B_sam_num, 1))
        with torch.no_grad():
            self.__Mean_Init()

    def __Mean_Init(self):
        """
        Line 1
        :return:
        """
        init.constant_(self.A_weight, 1.0)
        init.constant_(self.B_weight, 1.0)

    def __forward_alpha_beta(self):
        alpha = F.softmax(input=self.A_weight, dim=0)
        # alpha[alpha < self.eps] = 0.0
        beta = F.softmax(input=self.B_weight, dim=0)
        # beta[beta < self.eps] = 0.0
        return alpha, beta

    def __forward_A_B(self):
        """
        Line 10, 18
        :return:
        """
        alpha, beta = self.__forward_alpha_beta()
        a = torch.matmul(input=self.A, other=alpha)
        b = torch.matmul(input=self.B, other=beta)
        diff = a - b
        dis2 = torch.sum(input=diff * diff)
        return torch.relu(dis2)

    def __forward_K(self):
        """
        Line 8, 16
        :return:
        """
        alpha, beta = self.__forward_alpha_beta()
        alpha_beta = torch.cat(tensors=[alpha, beta], dim=0)
        dis2 = torch.matmul(input=alpha_beta.T, other=torch.matmul(input=self.K, other=alpha_beta))
        return torch.relu(dis2[0][0])

    def distance(self):
        """
        Line 7-11, 15-19
        :return:
        """
        if self.A is not None:
            return self.__forward_A_B()
        else:
            return self.__forward_K()

    def solving(self, max_ite_num=10000, min_ite_gap=1e-10, lr=1e-3, decay_how_often=None, decay_factor=0.5):
        """
        Algorithm 2
        :param max_ite_num:
        :param min_ite_gap:
        :param lr:
        :param decay_how_often:
        :param decay_factor:
        :return:
        """
        if decay_how_often is None:
            decay_how_often = max_ite_num
        max_ite_num = torch.tensor(data=max_ite_num, device=self.A_weight.device)
        ite_num = torch.tensor(data=0, device=self.A_weight.device)

        old_dis2 = torch.tensor(data=0, device=self.A_weight.device)
        new_dis2 = torch.tensor(data=np.inf, device=self.A_weight.device)
        min_dis2 = torch.tensor(data=np.inf, device=self.A_weight.device)

        optimizer = optim.Adam(params=self.parameters(), lr=lr)
        alpha = None
        beta = None
        while ite_num < max_ite_num and torch.abs(new_dis2 - old_dis2) > min_ite_gap:  # Line 12
            old_dis2 = new_dis2

            optimizer.zero_grad()
            new_dis2 = self.distance()
            new_dis2.backward()
            optimizer.step()  # Line 14

            new_dis2 = new_dis2.detach().clone()
            if new_dis2 < min_dis2:
                min_dis2 = new_dis2
                with torch.no_grad():
                    aa, bb = self.__forward_alpha_beta()
                    alpha = aa.detach().clone()
                    beta = bb.detach().clone()
            ite_num += 1

            if ite_num % decay_how_often == 0:
                Margin_DisConvexHulls.Decay_lr(optimizer=optimizer, decay_factor=decay_factor)

            # print('ite= ' + str(ite_num) + ' dis2= ' + str(new_dis2))

        return {'dis2': min_dis2.detach().clone().cpu().numpy(),
                'alpha': alpha,
                'beta': beta}

    @staticmethod
    def Decay_lr(optimizer: Optimizer, decay_factor=0.5):
        for para_group in optimizer.param_groups:
            para_group['lr'] = para_group['lr'] * decay_factor
