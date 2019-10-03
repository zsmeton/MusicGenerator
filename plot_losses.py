from keras.callbacks import Callback
from matplotlib import pyplot as plt
import json

class PlotLearning(Callback):
    def load_in_data(self, filename):
        data = []
        with open(filename, 'r') as fin:
            s = fin.readline()
            s = s.replace('"', '')
            s = s.replace("'", '"')
            self.logs = json.loads(s)

        for i,log in enumerate(self.logs):
            self.on_epoch_end(i, log)

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        print(logs)
        logs = dict([(key, [float(i) for i in value]) for key, value in logs.items()])
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('categorical_accuracy'))
        self.val_acc.append(logs.get('val_categorical_accuracy'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="validation loss")
        ax1.legend()

        ax2.plot(self.x, self.acc, label="categorical accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()

        plt.show()

    def on_train_end(self, logs={}):
        with open(f'logs.txt', 'w') as fout:
            json.dump(str(self.logs), fout)
