import random
import time
import torch
import hdf5storage
import numpy as np
from torch.backends import cudnn

from Core.VC_Dim_DNNBoostFramework import VC_Dim_DNNBoostFramework, MyFCNet3
from Core.DataSet import DataSet

"""
This demo shows how to use the VC Dimension based DNN Boosting Framework (given in Section 5).
Take Fully-Connected Network as an example.
See Section 6.3
"""

"""
0. Preparation stage
"""
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')

rand_seed = 123
torch.manual_seed(rand_seed)
torch.cuda.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)
random.seed(rand_seed)
np.random.seed(rand_seed)
cudnn.benchmark = True

data_dir = '/root/cjbDisk/Projects/GRLF_GPG/DataSet/MNIST/'
res_dir = '/root/cjbDisk/Projects/GRLF_GPG/Result/MNIST/'

VC_Dim_Loss_weight = 0.1
# When VC_Dim_Loss_weight>0, the proposed boosting module will work
# When VC_Dim_Loss_weight=0, the proposed boosting module will not work
lr = 0.0005
beg_str = time.strftime('%y-%m-%d@%H-%M-%S', time.localtime())
setting_str = beg_str + '@lr=' + str(lr) + '@VC_weight=' + str(VC_Dim_Loss_weight)


"""
1. Load MNIST training data set and test data set
"""
tr_data_set = DataSet(mat_data_file=data_dir + 'MNIST_train_100.mat')
te_data_set = DataSet(mat_data_file=data_dir + 'MNIST_test_10K.mat')

""" 
2. Construct Representation Learning Network and Classification Network
   and embed them into the proposed framework
"""
RLNet = MyFCNet3(in_features=28 * 28, out_features=256)
ClassificationNet = torch.nn.Linear(in_features=256, out_features=10)
mode = VC_Dim_DNNBoostFramework(RLNet=RLNet, ClassificationNet=ClassificationNet, VC_Dim_Loss_weight=VC_Dim_Loss_weight)
mode.to(device)

"""
3. End-to-End training
"""
mode.fit(train_data_set=tr_data_set, lr=lr, batch_size=1024, max_epoch_num=300, device=device,
         test_data_set=te_data_set, log_file_name=res_dir + setting_str + '@train_log.txt')

"""
4. Predicting """
res = mode.predict(test_data_set=te_data_set, device=device)

"""
5. Save the predicting results
"""
hdf5storage.savemat(file_name=res_dir + setting_str + '@predict_result.mat', mdict=res)
