from music21 import converter, instrument, note, chord, stream
import glob
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from plot_losses import PlotLearning
from user_input import get_user_yes_no, get_user_options
from tqdm import tqdm


def load_notes() -> list:
    notes = []
    song_files = glob.glob("./midi_songs/*.mid")
    for i, file in tqdm(enumerate(song_files), desc='Loading in songs'):
        midi = converter.parse(file)

        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)

        if parts:  # file has instrument parts
            parts = [part for part in parts if part.partName == 'Piano']
            if parts:
                notes_to_parse = parts[0].recurse()
        else:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
        if notes_to_parse:
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    print(element)
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes


def get_sequences(notes, sequence_length=90) -> (np.ndarray, np.ndarray, int, set):
    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    X = []
    y = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        X.append([note_to_int[char] for char in sequence_in])
        y.append(note_to_int[sequence_out])
    n_patterns = len(X)

    # reshape the input into a format compatible with LSTM layers
    X = np.reshape(X, (n_patterns, sequence_length, 1))

    # normalize input
    n_vocab = len(set(notes))  # get amount of pitch names
    print("n_vocab is: ",n_vocab)
    X = X / float(n_vocab)

    # one-hot code the output
    # for info on one-hot check out:
    #    https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
    y = keras.utils.to_categorical(y)
    return X, y, n_vocab, pitchnames


def create_model(X_shape, n_vocab) -> Sequential:
    lstm_model = Sequential()
    lstm_model.add(LSTM(1024, input_shape=X_shape, return_sequences=True, activation='relu'))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(LSTM(512, return_sequences=True, activation='relu'))
    lstm_model.add(Dropout(0.25))
    lstm_model.add(LSTM(256, activation='relu'))
    lstm_model.add(Dense(256))
    lstm_model.add(Dropout(0.1))
    lstm_model.add(Dense(n_vocab))
    lstm_model.add(Activation('softmax'))
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy', 'accuracy'])
    return lstm_model


def train_model(lstm_model: Sequential, X: np.ndarray, y: np.ndarray, loadpath='', epoch_start=0, epochs=1,
                validation_size=0.2):
    # Split data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, random_state=1)

    # Set up callbacks
    # Set when to checkpoint
    filepath = "lstm-weights.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
    # Set up live training plotting
    plot = PlotLearning()
    callbacks_list = [checkpoint, plot]

    # load weights if resume
    if loadpath:
        try:
            lstm_model.load_weights(loadpath)
        except OSError:
            raise ValueError(f"Invalid path to weights. '{loadpath}' not found.")

    # train the model

    train_history = lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                   epochs=epochs,
                                   initial_epoch=epoch_start, batch_size=2048,
                                   callbacks=callbacks_list, validation_freq=1, verbose=1)

def sample(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    probabilities = np.random.multinomial(1, predictions, 1)
    #print(predictions.flatten().shape, probabilities.shape, np.sum(predictions))
    # This return statement give us more variety, but it doesn't always sound great
    return np.random.choice(range(0, len(predictions.flatten())), p=predictions.flatten())
    # This return statement should be more predictable. On my computer, it only ever generates a single note, the training might have been off
    #return np.argmax(probabilities)

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
        #pick note with likelyhood predicted rather than max

        index = sample(prediction, temperature=1.1)

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
    # Model file
    model_file = "lstm-weights.hdf5"
    #  load in data
    X, y, n_vocab = None, None, None
    if not glob.glob("X.npy") or not glob.glob("Y.npy") or not glob.glob("vocab.npy"):
        notes = load_notes()
        X, y, n_vocab, pitch_names = get_sequences(notes)
        np.save("X", X)
        np.save("Y", y)
        np.save("vocab", np.array([n_vocab]))
        n_vocab = np.array([n_vocab])

    else:
        print("Loading data from saved numpy files:", end='')
        X = np.load("X.npy")
        print("..", end='')
        y = np.load("Y.npy")
        print("..", end='')
        n_vocab = np.load("vocab.npy")
        print("..", end='')
        print("Done")

    option = get_user_options('What would you like to do:',['Train the model', 'Generate music', 'Create a picture of the model', 'Exit'])
    while option < 3:
        if option == 0:
            # create model
            model = create_model((X.shape[1], X.shape[2]), n_vocab[0])
            # train the model
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
            train_model(model, X, y)
        elif option == 1:
            loaded_model = create_model((X.shape[1], X.shape[2]), n_vocab[0])
            loaded_model.load_weights(model_file)
            file_name = input('Song(file) name:')
            generate_music(loaded_model, save_file=file_name)
        elif option == 2:
            loaded_model = create_model((X.shape[1], X.shape[2]), n_vocab[0])
            loaded_model.load_weights(model_file)
            plot_model(loaded_model, to_file='model.svg', show_shapes=True, expand_nested=True)

        option = get_user_options('What would you like to do:',
                                  ['Train the model', 'Generate music', 'Create a picture of the model', 'Exit'])