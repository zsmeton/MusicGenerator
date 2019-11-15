import random

from keras import Sequential, optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense, Dropout, LSTM, SimpleRNN
from sklearn.model_selection import train_test_split

from Generator import My_Custom_Generator
from plot_losses import PlotLearning
from prep_batch_loading import read_size_of_data
from training_utils import *
from user_input import get_user_options, get_user_non_negative_number_or_default, get_user_yes_no, \
    get_user_non_negative_number, get_user_filename


def create_model_duration(X_shape) -> Sequential:
    lstm_model = Sequential()
    lstm_model.add(LSTM(
        256,
        input_shape=X_shape,
        return_sequences=True,
        activation='sigmoid', kernel_initializer='he_uniform'
    ))
    lstm_model.add(SimpleRNN(512, activation='sigmoid', kernel_initializer='he_uniform'))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(256, activation='sigmoid', kernel_initializer='he_uniform'))
    lstm_model.add(Dense(1, activation='sigmoid', kernel_initializer='he_uniform'))
    lstm_model.compile(loss='mse', optimizer='adam', metrics=[rmse])
    return lstm_model


def train_model_duration(lstm_model: Sequential, epochs=200, initial_epoch=0):

    # Set up callbacks
    # Set when to checkpoint
    filepath = "models/duration/duration-model-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=False, mode='min')

    # Set up live training plotting
    plot = PlotLearning('rmse', 'root mean squared error', 'models/duration/duration_logs.txt', 'models/duration/graph_duration')
    callbacks_list = [checkpoint, plot]

    # set up training plotting
    plot.on_train_begin()
    if glob.glob('models/duration/duration_logs.txt') and initial_epoch != 0:
        plot.load_in_data('models/duration/duration_logs.txt')

    train_batch_size = 1
    train_x_files = glob.glob('batch_data/duration/train/x*')
    train_y_files = glob.glob('batch_data/duration/train/y*')
    if len(train_x_files) != len(train_y_files):
        raise FileExistsError("The number of x and y values for training is not the same")
    my_training_batch_generator = My_Custom_Generator(train_x_files, train_y_files, train_batch_size)

    val_batch_size = 1
    val_x_files = glob.glob('batch_data/duration/val/x*')
    val_y_files = glob.glob('batch_data/duration/val/y*')
    if len(val_x_files) != len(val_y_files):
        raise FileExistsError("The number of x and y values for validation is not the same")
    my_validation_batch_generator = My_Custom_Generator(val_x_files, val_y_files, val_batch_size)

    lstm_model.fit_generator(generator=my_training_batch_generator,
                        steps_per_epoch=len(my_training_batch_generator),
                        epochs=epochs,
                        initial_epoch=initial_epoch,
                        verbose=1,
                        validation_data=my_validation_batch_generator,
                        validation_steps=len(my_validation_batch_generator), callbacks=callbacks_list)


if __name__ == '__main__':

    option = get_user_options('What would you like to do', ['Train the model', 'Exit'])
    while option < 2:
        if option == 1:

            # Build the model
            model = create_model_duration(read_size_of_data())
            model.summary()

            if get_user_yes_no('Would you like to resume a training session'):
                start_epoch = int(get_user_non_negative_number('What epoch were you on'))
                end_epoch = int(get_user_non_negative_number('How many epochs would you like to train in total'))
                filename = get_user_filename("What is the model weight file")
                model.load_weights(filename)
                # train the model
                train_model_duration(model, epochs=end_epoch, initial_epoch=start_epoch)
            else:
                end_epoch = int(get_user_non_negative_number('How many epochs would you like to run'))
                train_model_duration(model, epochs=end_epoch)

        option = get_user_options('What would you like to do',
                                  ['Train the model', 'Exit'])
