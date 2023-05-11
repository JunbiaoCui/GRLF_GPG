import numpy as np
import hdf5storage


# Python numpy based DataSet

class DataSet(object):
    def __init__(self, mat_data_file):
        Xy = hdf5storage.loadmat(file_name=mat_data_file)
        self.y = np.squeeze(Xy['y']).astype(np.int64)
        self.X = Xy['X'].astype(np.float64)

        self.__rand_sam_ind = None
        self.__batch_beg_ind = None
        self.__batch_size = None

    def Feature_Num(self):
        return self.X.shape[1]

    def Class_Num(self):
        return np.unique(self.y).shape[0]

    def Init_Epoch(self, epoch_num=1, batch_size=32, do_shuffle=True):
        sam_num = self.y.shape[0]
        self.__rand_sam_ind = np.array(list(range(sam_num)) * epoch_num)
        if do_shuffle:
            np.random.shuffle(self.__rand_sam_ind)
        self.__batch_beg_ind = 0
        self.__batch_size = batch_size

    def Next_Batch(self):
        batch_end_ind = self.__batch_beg_ind + self.__batch_size
        if batch_end_ind > self.__rand_sam_ind.shape[0]:
            batch_end_ind = self.__rand_sam_ind.shape[0]
        batch_sam_ind = self.__rand_sam_ind[self.__batch_beg_ind: batch_end_ind]
        self.__batch_beg_ind = batch_end_ind
        return {'batch_X': self.X[batch_sam_ind],
                'batch_y': self.y[batch_sam_ind],
                'is_last_batch': batch_end_ind == self.__rand_sam_ind.shape[0]}

    def All_X_y(self):
        return {'X': self.X, 'y': self.y}


class Image_DataSet(DataSet):
    def __init__(self, mat_data_file, image_channel, image_width, image_high):
        super(Image_DataSet, self).__init__(mat_data_file)
        self.X = np.reshape(a=self.X, newshape=[self.y.shape[0],
                                                image_channel,
                                                image_width,
                                                image_high])
