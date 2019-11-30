from keras import Sequential, optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense, Dropout, LSTM,  RepeatVector, TimeDistributed, BatchNormalization
from sklearn.model_selection import train_test_split

from multioutput.Generator import My_Custom_Generator
from multioutput.plot_losses import PlotLearning
from prep_batch_loading import read_size_of_data, weighted_binary_crossentropy, glob
import numpy as np


def create_note_generator(X_shape, out_length) -> Sequential:
    lstm_model = Sequential()
    lstm_model.add(LSTM(
        input_shape=X_shape,
        return_sequences=False,
        output_dim=256,
        activation='tanh',
        dropout=0.1,
        recurrent_dropout=0.1
    ))
    lstm_model.add(RepeatVector(out_length))
    lstm_model.add(TimeDistributed(BatchNormalization(momentum=0.7)))
    lstm_model.add(LSTM(units=256,
                        return_sequences=True,
                        activation='tanh',
                        dropout=0.1,
                        recurrent_dropout=0.1))
    lstm_model.add(TimeDistributed(BatchNormalization(momentum=0.7)))
    lstm_model.add(TimeDistributed(Dense(256, activation='relu')))
    lstm_model.add(TimeDistributed(Dense(128, activation='sigmoid')))
    return lstm_model


if __name__ == "__main__":
    note_generator = create_note_generator(read_size_of_data(), 50)
    note_generator.summary()
    note_generator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Set up callbacks
    # Set when to checkpoint
    filepath = "models/notes/note-model-{epoch:02d}-{accuracy:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=0, save_best_only=False, mode='max')

    # Set up live training plotting
    plot = PlotLearning('accuracy', 'accuracy', 'models/notes/notes_logs.txt', 'models/notes/graph_notes')
    callbacks_list = [checkpoint, plot]

    # set up training plotting
    plot.on_train_begin()


    train_batch_size = 1
    train_x_files = glob.glob('batch_data/notes/train/x*')
    train_y_files = glob.glob('batch_data/notes/train/y*')
    if len(train_x_files) != len(train_y_files):
        raise FileExistsError("The number of x and y values for training is not the same")
    my_training_batch_generator = My_Custom_Generator(train_x_files, train_y_files, train_batch_size)

    val_batch_size = 1
    val_x_files = glob.glob('batch_data/notes/val/x*')
    val_y_files = glob.glob('batch_data/notes/val/y*')
    if len(val_x_files) != len(val_y_files):
        raise FileExistsError("The number of x and y values for validation is not the same")
    my_validation_batch_generator = My_Custom_Generator(val_x_files, val_y_files, val_batch_size)

    note_generator.fit_generator(generator=my_training_batch_generator,
                             steps_per_epoch=len(my_training_batch_generator),
                             epochs=1,
                             verbose=1,
                             validation_data=my_validation_batch_generator,
                             validation_steps=len(my_validation_batch_generator), callbacks=callbacks_list)

    # note_generator.load_weights("models/notes/note-model-01-0.9830.hdf5")

    x = np.load("batch_data/notes/val/x0.npy", allow_pickle=True)
    y = note_generator.predict(np.array([x[0]]))
    print(y)
