import numpy as np
import torch
from scipy.spatial import distance

from Core.Radius import Radius
from Core.Margin_DisConvexHulls import Margin_DisConvexHulls

"""
Dim: Dimension

sam: Sample

See Section 4

See Algorithm 3 VC Dimension based Kernel Selection in Appendix B.2
"""


class VC_Dim_KernelSelection(object):
    @staticmethod
    def work(X_pos: np.ndarray, X_neg: np.ndarray, Gaussian_kernel_para_arr: np.ndarray, device=torch.device('cpu')):
        """

        :param X_pos: shape=[n_pos_sam, n_feature]
        :param X_neg: shape=[n_neg_sam, n_feature]
        :param Gaussian_kernel_para_arr:
        :param device:
        :return:
        """
        num_pos_sam = X_pos.shape[0]
        num_neg_sam = X_neg.shape[0]
        n_sam = num_pos_sam + num_neg_sam

        X_pn = np.concatenate((X_pos, X_neg), axis=0)
        D = distance.pdist(X=X_pn, metric='euclidean')
        D = distance.squareform(D * D)
        # D[i,j]= ||X[i,:]-X[j,:] ||_2^2

        num_kernel = Gaussian_kernel_para_arr.shape[0]
        kernel_score_arr = np.ones(shape=[num_kernel])
        for i in range(num_kernel):
            # print('i=' + str(i))
            para = Gaussian_kernel_para_arr[i]
            K = np.exp(para * D)   # K[i,j]=exp(para*D[i,j])
            K = torch.tensor(data=K, device=device)

            # Invoke Algorithm 1 in Appendix B.1
            radius = Radius(sample_num=n_sam, X=None, K=K)
            radius.to(device)
            res = radius.solving(max_ite_num=10000, min_ite_gap=1e-10, lr=1e-3, decay_how_often=None, decay_factor=0.95)
            R2 = res['R2']  # np.array

            # Invoke Algorithm 2 in Appendix B.1
            margin = Margin_DisConvexHulls(A_sam_num=num_pos_sam, B_sam_num=num_neg_sam, A=None, B=None, K=K)
            margin.to(device)
            res = margin.solving(max_ite_num=10000, min_ite_gap=1e-10, lr=1e-3, decay_how_often=None, decay_factor=0.95)
            M2 = res['dis2']  # np.array

            val = 4 * R2 / M2  # Algorithm 3 Line 6
            kernel_score_arr[i] = val

        best_ind = np.argmin(kernel_score_arr)  # Algorithm 3 Lines 7-10

        return {'best_kernel_para': Gaussian_kernel_para_arr[best_ind],
                'Gaussian_kernel_para_arr': Gaussian_kernel_para_arr,
                'kernel_score_arr': kernel_score_arr}
