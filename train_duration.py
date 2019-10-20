import random

from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split

from plot_losses import PlotLearning
from training_utils import *
from user_input import get_user_options, get_user_non_negative_number_or_default, get_user_yes_no, \
    get_user_non_negative_number, get_user_filename


def create_model_duration(X_shape) -> Sequential:
    lstm_model = Sequential()
    lstm_model.add(LSTM(
        256,
        input_shape=X_shape,
        return_sequences=True, activation='tanh'
    ))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(512, return_sequences=True, activation='tanh'))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(256, activation='tanh'))
    lstm_model.add(Dense(256))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(256))
    lstm_model.add(Dropout(0.3))
    lstm_model.add(Dense(1))
    lstm_model.add(Activation('relu'))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam', metrics=[rmse])
    return lstm_model


def train_model_durations(lstm_model: Sequential, X: np.ndarray, sequence_length, X_val=None, epochs=200, initial_epoch=0,
                          validation_size=0.2, songs_per_epoch=10):
    # Split data into train and test data
    if X_val is None:
        X, X_val = train_test_split(X, test_size=validation_size, random_state=1)

    X_val_seq, y_val = sequence_songs(X_val, sequence_length,sequence_method='duration', data_multiplier=sequence_length//3)

    # Set up callbacks
    # Set when to checkpoint
    filepath = "duration-model-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
    # Set up live training plotting
    plot = PlotLearning('rmse', 'root mean squared error', 'duration_logs.txt')
    callbacks_list = [checkpoint]

    # set up training plotting
    plot.on_train_begin()
    if glob.glob('duration_logs.txt') and initial_epoch != 0:
        plot.load_in_data('duration_logs.txt')

    # train the model
    keep_data = 2
    try:
        for i in range(initial_epoch, epochs,keep_data):
            # choose training song
            song_index = i + random.randrange(keep_data)

            # If we have gone through all the songs, randomize the list and try again
            if song_index > (len(X) - songs_per_epoch):
                X = randomize(X, 3 * len(X))
                song_index %= len(X) - songs_per_epoch

            X_seq, y = sequence_songs(X, sequence_length, sequence_method='duration', size_of_array=1500, starting_position=song_index)

            for j in range(keep_data):
                train_history = lstm_model.fit(X_seq, y, validation_data=(X_val_seq, y_val),
                                               epochs=i+j+1, initial_epoch=i+j, batch_size=64,
                                               callbacks=callbacks_list, validation_freq=1, verbose=1)
                plot.on_epoch_end(train_history.epoch, train_history.history)

            plot.on_train_end('duration_training_graph.png')
    except:
        plot.on_train_end('duration_training_graph.png')
        raise


if __name__ == '__main__':

    option = get_user_options('What would you like to do',['Train the model', 'Exit'])
    while option < 2:
        if option == 1:
            # Load in the data
            X_train, X_val = getX_train_val()

            sequence_length = 70
            # create model
            model = create_model_duration((sequence_length, X_train[0].shape[1]))

            if get_user_yes_no('Would you like to resume a training session'):
                start_epoch = int(get_user_non_negative_number('What epoch were you on'))
                end_epoch = int(get_user_non_negative_number('How many epochs would you like to train in total'))
                filename = get_user_filename("What is the model weight file")
                model.load_weights(filename)
                # train the model
                train_model_durations(model, X_train, sequence_length, X_val=X_val, epochs=end_epoch, songs_per_epoch=7, initial_epoch=start_epoch)
            else:
                end_epoch = int(get_user_non_negative_number('How many epochs would you like to run'))
                train_model_durations(model, X_train, sequence_length, X_val=X_val, epochs=end_epoch, songs_per_epoch=7)

        option = get_user_options('What would you like to do:',
                                  ['Train the model', 'Exit'])
