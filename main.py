from music21 import converter, instrument, note, chord, stream
import glob
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from plot_losses import PlotLearning
from user_input import get_user_yes_no, get_user_options, get_user_non_negative_number
from tqdm import tqdm
from guppy import hpy


def load_song(song_path:str):
    midi = converter.parse(song_path)
    notes_to_parse = None
    parts = instrument.partitionByInstrument(midi)

    # Create 1 x 89 np array
    thisSlice = np.zeros(129, dtype=float)
    lastOffset = 0.0
    value = 0.0
    # create notes to append to
    notes = []
    if parts:  # file has instrument parts
        parts = [part for part in parts if part.partName == 'Piano']
        if parts:
            notes_to_parse = parts[0].recurse()
    else:  # file has notes in a flat structure
        notes_to_parse = midi.flat.notes
    if notes_to_parse:
        for element in notes_to_parse:
            if lastOffset != element.offset:
                # This gets me when the note is played
                # element.offset
                value = element.offset - lastOffset
                lastOffset = element.offset
            if isinstance(element, note.Note):
                # This gets me the note played in midi
                # element.pitch.midi
                thisSlice[element.pitch.midi - 1] = 1
            elif isinstance(element, chord.Chord):
                for n in element.notes:
                    thisSlice[n.pitch.midi - 1] = 1
            thisSlice[128] = value
            notes.append(thisSlice)
            thisSlice = np.zeros(129, dtype=float)
    return notes


def load_notes(folder) -> list:
    notes = []
    # Change songs loaded later, took way too long on my desktop
    song_files = glob.glob(f"./{folder}/*.mid")
    for i, file in tqdm(enumerate(song_files), desc='Loading in songs'):
        notes.append(np.array(load_song(file)))
    return notes


def get_sequences(notes, sequence_length=70, data_multiplier=4, verbose=True) -> (np.array, np.array):
    X = []
    y = []

    # create input sequences and the corresponding outputs
    for i in tqdm(range(0, len(notes) - sequence_length, sequence_length//data_multiplier), desc='Segmenting songs into sequences', disable=(not verbose)):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        X.append(sequence_in)
        y.append(sequence_out)

    return np.array(X), np.array(y)


def sequence_songs(X, sequence_length, data_multiplier=2):
    X_seq = []
    y = []
    for song in X:
        x_, y_ = get_sequences(song, sequence_length, data_multiplier=data_multiplier, verbose=False)
        X_seq.extend(x_)
        y.extend(y_)
    X_seq = np.array(X_seq)
    y = np.array(y)
    return X_seq, y


def create_model(X_shape) -> Sequential:
    lstm_model = Sequential()
    lstm_model.add(LSTM(
        256,
        input_shape=X_shape,
        return_sequences=True, activation='tanh'
    ))
    lstm_model.add(Dropout(0.2))
    #lstm_model.add(LSTM(512, return_sequences=True, activation='tanh'))
    #lstm_model.add(Dropout(0.2))
    #lstm_model.add(LSTM(256, return_sequences=True, activation='tanh'))
    #lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(256, activation='tanh'))
    lstm_model.add(Dense(256))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(256))
    lstm_model.add(Dropout(0.3))
    lstm_model.add(Dense(129))
    lstm_model.add(Activation('tanh'))
    lstm_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    return lstm_model


def train_model(lstm_model: Sequential, X: np.ndarray, sequence_length, X_val=None, epochs=200, initial_epoch=0,
                validation_size=0.2, songs_per_epoch=10):
    # Split data into train and test data
    if X_val is None:
        X, X_val = train_test_split(X, test_size=validation_size, random_state=1)

    X_val_seq, y_val = sequence_songs(X_val, sequence_length, sequence_length//3)

    # Set up callbacks
    # Set when to checkpoint
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
    # Set up live training plotting
    plot = PlotLearning()
    callbacks_list = [checkpoint]

    # set up training plotting
    plot.on_train_begin()
    if glob.glob('logs.txt') and initial_epoch != 0:
        plot.load_in_data('logs.txt')

    # train the model
    try:
        for i in range(initial_epoch, epochs):
            song_index = i % (len(X)-songs_per_epoch)  # choose training song
            X_seq, y = sequence_songs(X[song_index:i+songs_per_epoch], sequence_length, data_multiplier=sequence_length)
            train_history = lstm_model.fit(X_seq, y, validation_data=(X_val_seq, y_val),
                                           epochs=i+1, initial_epoch=i, batch_size=64,
                                           callbacks=callbacks_list, validation_freq=1, verbose=1)
            plot.on_epoch_end(train_history.epoch, train_history.history)
    except:
        plot.on_train_end()
        raise


def generate_music(l_model, starter_notes=30, save_file='test_output'):
    notes = load_notes()
    start = np.random.randint(0, len(X) - (starter_notes+1))
    int_to_note = dict((number, note) for number, note in enumerate(sorted(set(item for item in notes))))
    pattern = list(X[start])
    prediction_output = []
    # Run on some starter data
    for i in tqdm(range(30), desc='Seeding music generation'):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab[0])
        prediction = l_model.predict(prediction_input, verbose=0)
        pattern = list(X[start+i])

    # generate 500 notes
    pattern = list(X[start])
    for note_index in tqdm(range(500), desc='Generating music'):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab[0])
        prediction = l_model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    offset = 0
    output_notes = []
    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=save_file+'.mid')


if __name__ == '__main__':
    h = hpy() # can call print(h.heap()) to view current heap usage
    X_train = load_notes('midi_songs/training')
    X_val = load_notes('midi_songs/validation')

    option = get_user_options('What would you like to do:',['Train the model', 'Generate music', 'Create a picture of the model', 'Exit'])
    while option < 3:
        if option == 0:
            sequence_length = 70
            # create model
            model = create_model((sequence_length, X_train[0].shape[1]))

            if get_user_yes_no('Would you like to resume a training session'):
                filename = ''
                while not glob.glob(filename):
                    filename = input("What is the name of the file:")
                model.load_weights(filename)
                start_epoch = get_user_non_negative_number('What epoch were you on')
                end_epoch = get_user_non_negative_number('How many epochs would you like to train in total')
                # train the model
                train_model(model, X_train, sequence_length, X_val=X_val, epochs=end_epoch, songs_per_epoch=10, initial_epoch=start_epoch)
            else:
                end_epoch = get_user_non_negative_number('How many epochs would you like to run')
                train_model(model, X_train, sequence_length, X_val=X_val, epochs=end_epoch, songs_per_epoch=10)

        elif option == 1:
            print("NOT IMPLEMENTED")
            assert False
        elif option == 2:
            print("NOT IMPLEMENTED")
            assert False

        option = get_user_options('What would you like to do:',
                                  ['Train the model', 'Generate music', 'Create a picture of the model', 'Exit'])
