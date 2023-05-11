import torch
from torch import Tensor, nn

from Core.Margin_DisConvexHulls import Margin_DisConvexHulls

"""
VC Dimension based Loss

See formula (14)

See Algorithm 4 VC Dimension based DNN Boosting Framework (Lines 4-10) in Appendix B.3

See Figure 1
"""


class VC_DimLoss(nn.Module):
    def __init__(self):
        super(VC_DimLoss, self).__init__()
        self.dis_ch_loss = DisConvexHulls_Loss()

    def forward(self, X: Tensor, y: Tensor):
        """

        :param X: shape=[n_sam, n_feature], double
        :param y: shape=[n_sam], int
        :return:
       """
        return self.dis_ch_loss(X=X, y=y)


"""
Distance between 
Convex Hull of Positive Samples
and
Convex Hull of Negative Samples
"""


class DisConvexHulls_Loss(nn.Module):
    def __init__(self):
        super(DisConvexHulls_Loss, self).__init__()

    def forward(self, X: Tensor, y: Tensor):
        """

        :param X: shape=[n_sam, n_feature], double
        :param y: shape=[n_sam], int
        :return:
        """
        dis_list = []
        class_space = torch.unique(y)
        class_num = class_space.shape[0]
        """ One vs One, Invoke Algorithm 2 """
        for i in range(class_num):
            X_pos = X[y == class_space[i]]
            for j in range(i + 1, class_num):
                # print('(' + str(i) + ' vs ' + str(j) + ')')
                X_neg = X[y == class_space[j]]
                dis_ch = Margin_DisConvexHulls(A_sam_num=X_pos.shape[0],
                                               B_sam_num=X_neg.shape[0],
                                               A=X_pos.detach().clone(),
                                               B=X_neg.detach().clone())
                dis_ch.to(X.device)
                res = dis_ch.solving(max_ite_num=100, min_ite_gap=1e-10,
                                     decay_how_often=10, decay_factor=0.9)
                alpha = res['alpha'].detach().clone()
                beta = res['beta'].detach().clone()
                p = torch.matmul(input=X_pos.T, other=alpha)
                n = torch.matmul(input=X_neg.T, other=beta)
                diff = p - n
                dis = torch.sum(input=diff * diff)
                dis_list.append(dis.view([-1]))
        loss = torch.cat(tensors=dis_list, dim=0)
        return - torch.mean(input=loss)
