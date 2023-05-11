import time
import numpy as np
import torch
from torch import nn, optim, Tensor

from Core.VC_Dim_Loss import VC_DimLoss
from Core.StandNet import StandNet
from Core.DataSet import DataSet

"""
Dim: Dimension

DNN: Deep Neural Network

RLNet: Representation Learning Network

ClassificationNet: Classification Network

See Section 5

The proposed boosting module is marked with a solid red box in Figure 1 

See Algorithm 4 VC Dimension based DNN Boosting Framework in Appendix B.3
"""


class VC_Dim_DNNBoostFramework(nn.Module):
    def __init__(self, RLNet: nn.Module, ClassificationNet: nn.Module, VC_Dim_Loss_weight):
        """

        :param RLNet:
        :param ClassificationNet:
        :param VC_Dim_Loss_weight: >=0, trade-off parameter
        When VC_Dim_Loss_weight>0, the proposed boosting module will work
        When VC_Dim_Loss_weight=0, the proposed boosting module will not work
        """
        super(VC_Dim_DNNBoostFramework, self).__init__()
        self.RLNet = RLNet
        self.ClassificationNet = ClassificationNet
        self.ce_loss = nn.CrossEntropyLoss()
        self.VC_Dim_Loss_weight = VC_Dim_Loss_weight

        self.std_net = None
        self.VC_Dim_loss = None
        if self.VC_Dim_Loss_weight > 0:
            self.std_net = StandNet(p=2)
            self.VC_Dim_loss = VC_DimLoss()

        self.optimizer = None

    def __forward(self, X):
        """

        :param X: shape=[n_sam, n_feature]
        :return:
        """
        H = self.RLNet(X)

        if self.VC_Dim_Loss_weight > 0:
            H = self.std_net(H)
        else:
            H = torch.relu(H)

        Y = self.ClassificationNet(H)

        return {'H': H, 'Y': Y}

    def __update_parameters(self, batch_data, device=torch.device('cpu')):
        self.optimizer.zero_grad()
        batch_X = torch.tensor(data=batch_data['batch_X'], dtype=torch.float64, device=device)
        batch_y = torch.tensor(data=batch_data['batch_y'], dtype=torch.int64, device=device)
        batch_pre = self.__forward(X=batch_X)

        batch_ce_loss = self.ce_loss(input=batch_pre['Y'], target=batch_y)
        batch_loss = batch_ce_loss
        batch_vc_loss = None
        if self.VC_Dim_Loss_weight > 0:
            batch_vc_loss = self.VC_Dim_loss(X=batch_pre['H'], y=batch_y)
            batch_loss = batch_loss + self.VC_Dim_Loss_weight * batch_vc_loss

        batch_loss.backward()
        self.optimizer.step()

        res = {'ce_loss': batch_ce_loss.clone().detach().cpu().numpy(),
               'vc_loss': 'None'}
        if batch_vc_loss is not None:
            res['vc_loss'] = batch_vc_loss.clone().detach().cpu().numpy()
        return res

    def fit(self, train_data_set: DataSet, lr=1e-3, batch_size=32,
            max_epoch_num=10000, device=torch.device('cpu'),
            test_data_set=None, log_file_name=None):
        log_write = None
        if log_file_name is not None:
            log_write = open(file=log_file_name, mode='w')

        self.train()
        self.optimizer = optim.Adam(params=self.parameters(), lr=lr)
        epoch_id = 0
        while epoch_id < max_epoch_num:
            train_data_set.Init_Epoch(batch_size=batch_size, do_shuffle=True)
            loss = None
            while True:
                batch_data = train_data_set.Next_Batch()
                loss = self.__update_parameters(batch_data=batch_data, device=device)
                if batch_data['is_last_batch']:
                    break

            acc = -1
            if test_data_set is not None:
                acc = self.predict(test_data_set=test_data_set, device=device)['acc']
            epoch_id += 1

            print(time.strftime("%y-%m-%d@%H:%M:%S", time.localtime()) +
                  ' epoch ' + str(epoch_id) + ' end, ce_loss= ' + str(loss['ce_loss']) +
                  ' vc_loss= ' + str(loss['vc_loss']) +
                  ' acc= ' + str(acc))
            if log_write is not None:
                log_write.write(time.strftime("%y-%m-%d@%H:%M:%S", time.localtime()) + '\t' +
                                '\tepoch\t' + str(epoch_id) +
                                '\tend.\tce_loss=\t' + str(loss['ce_loss']) +
                                '\tvc_loss=\t' + str(loss['vc_loss']) +
                                '\tacc=\t' + str(acc) + '\n')
                log_write.flush()
        if log_write is not None:
            log_write.close()

    def predict(self, test_data_set: DataSet, batch_size=1024, device=torch.device('cpu')):
        self.eval()
        test_data_set.Init_Epoch(epoch_num=1, batch_size=batch_size, do_shuffle=False)
        pre_y = []
        while True:
            batch_data = test_data_set.Next_Batch()
            batch_X = torch.tensor(data=batch_data['batch_X'], dtype=torch.float64, device=device)
            with torch.no_grad():
                batch_pre = self.__forward(X=batch_X)
                batch_pre_y = torch.argmax(input=batch_pre['Y'], dim=1)
            pre_y.append(batch_pre_y.detach().clone().cpu().numpy())
            if batch_data['is_last_batch']:
                break
        pre_y = np.concatenate(pre_y, axis=0)
        test_y = test_data_set.All_X_y()['y']
        test_acc = np.equal(pre_y, test_y) + 0.0
        test_acc = np.mean(test_acc)
        return {'pre_y': pre_y, 'acc': test_acc}


"""
FCNet: Fully-Connected Network
"""


class MyFCNet3(nn.Module):

    def __init__(self, in_features, out_features):
        super(MyFCNet3, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=out_features),
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.net(X)
