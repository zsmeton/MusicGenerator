from keras.callbacks import Callback
from matplotlib import pyplot as plt
import json


class PlotLearning(Callback):
    def __init__(self, metric, metric_desc, log_file):
        self.metric = metric
        self.metric_desc = metric_desc
        self.log_file = log_file

    def load_in_data(self, filename):
        logs = None
        with open(filename, 'r') as fin:
            s = fin.readline()
            s = s.replace('"', '')
            s = s.replace("'", '"')
            logs = json.loads(s)

        if logs is not None:
            for i, log in enumerate(logs):
                self.on_epoch_end(i, log, False)

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}, show=True):
        logs = dict([(key, [float(i) for i in value]) for key, value in logs.items()])
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get(f'{self.metric}'))
        self.val_acc.append(logs.get(f'val_{self.metric}'))

        self.i += 1
        if show:
            f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

            ax1.plot(self.x, self.losses, label="loss")
            ax1.plot(self.x, self.val_losses, label="validation loss")
            ax1.legend()

            ax2.plot(self.x, self.acc, label=f"{self.metric_desc}")
            ax2.plot(self.x, self.val_acc, label=f"validation {self.metric_desc}")
            ax2.plot(self.val_acc.index(min(self.val_acc)), min(self.val_acc), )
            ax2.legend()

            plt.show()

    def on_train_end(self, logs={}, savefile='plot.png'):
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="validation loss")
        ax1.legend()

        ax2.plot(self.x, self.acc, label=f"{self.metric_desc}")
        ax2.plot(self.x, self.val_acc, label=f"validation {self.metric_desc}")
        ax2.plot(self.val_acc.index(min(self.val_acc)), min(self.val_acc), )
        ax2.legend()

        plt.savefig(savefile)

        with open(f'{self.log_file}', 'w') as fout:
            json.dump(str(self.logs), fout)
