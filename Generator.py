import keras
import numpy as np


class My_Custom_Generator(keras.utils.Sequence):
    def __init__(self, x_filenames, label_filenames, batch_size):
        self.x_filenames = x_filenames
        self.label_filenames = label_filenames
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.x_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.x_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.label_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]

        X = []
        y = []
        for file_x, file_y in zip(batch_x, batch_y):
            X.extend(np.load(file_x, allow_pickle=True))
            y.extend(np.load(file_y, allow_pickle=True))

        return X,y