import time
import numpy as np
import hdf5storage
import torch

from Core.VC_Dim_KernelSelection import VC_Dim_KernelSelection

"""
This demo shows how to use the VC Dimension based Kernel Selection Method (given in Section 4).
Take Gaussian kernel as an example.
Taichi data set, See Appendix C.1
See Section 6.1
"""

"""
0. Preparation stage
"""
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:1')

data_dir = '/root/cjbDisk/Projects/GRLF_GPG/DataSet/TaiChi/'
res_dir = '/root/cjbDisk/Projects/GRLF_GPG/Result/TaiChi/'
beg_str = time.strftime('%y-%m-%d@%H-%M-%S', time.localtime())

"""
1. Load Taichi training data set
"""
Xy = hdf5storage.loadmat(data_dir + 'TaiChi_train.mat')
X = Xy['X'].astype(np.float64)
y = np.squeeze(Xy['y']).astype(np.int64)
X_pos = X[y == 1]
X_neg = X[y == -1]

"""
2. Generate candidate Gaussian kernel parameters
"""
para_arr = np.linspace(start=-200, stop=-1e-5, num=2000, endpoint=True)

"""
3. Invoke Algorithm 3 VC Dimension based Kernel Selection in Appendix B.2
"""
res = VC_Dim_KernelSelection.work(X_pos=X_pos, X_neg=X_neg, Gaussian_kernel_para_arr=para_arr, device=device)

"""
4. Save result
"""
hdf5storage.savemat(file_name=res_dir + beg_str + '@kernel_selection_result.mat', mdict=res)
print(res['best_kernel_para'])
