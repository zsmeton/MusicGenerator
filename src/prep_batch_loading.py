from sklearn.model_selection import train_test_split

import os

from multiprocessing import freeze_support, Value, Lock, Process
import keras
import music21
from music21 import converter, instrument, note, chord
import glob
import numpy as np
from tqdm import tqdm
from include.user_input import get_user_yes_no


def save_size_of_data(size):
    with open('files/batch_data/shape.txt', 'w') as fout:
        fout.write(str(size))


def read_size_of_data():
    with open('files/batch_data/shape.txt', 'r') as fin:
        return eval(fin.readline())


def save_pitchnames(pitchnames):
    with open('files/batch_data/pitchnames.txt', 'w') as fout:
        fout.write(str(pitchnames))


def read_pitchnames():
    with open('files/batch_data/pitchnames.txt', 'r') as fin:
        return sorted(set(eval(fin.readline())))


def delete_dir_contents(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def size_of_numpy_array(arr: np.array):
    return arr.size * arr.dtype.itemsize


def load_notes(path):
    songs = []
    song_files = glob.glob(f"{path}/*.mid")
    for i, file in tqdm(enumerate(song_files), desc='Loading in songs'):
        notes = []
        try:
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
                        notes.append('.'.join(str(n) for n in element.normalOrder))
            songs.append(np.array(notes))

        except music21.midi.MidiException as e:
            print(f"[ERROR]: Bad MIDI file, please remove {file}")
            print(e)

    # get all pitch names
    all_notes = [item for sublist in songs for item in sublist]
    pitchnames = sorted(set(all_notes))
    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number+1) for number, note in enumerate(pitchnames))
    n_vocab = len(pitchnames+1)  # get amount of pitch names

    return np.array(songs), pitchnames, note_to_int, n_vocab


def get_sequences(notes, sequence_length, note_to_int, n_vocab) -> (np.ndarray, np.ndarray, int, set):
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
    X = np.reshape(X, (n_patterns, 1, sequence_length))

    # one-hot code the output
    # for info on one-hot check out:
    #    https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
    y = keras.utils.to_categorical(y, n_vocab)
    return X, y


def threaded_sequencing(i, X, sequence_length, note_to_int, n_vocab, size_of_array, save_path, counter: Value, locker: Lock):
    X_seq = []
    y = []
    current_size = 0

    # Chunk sequences into n sequences each of size of array
    x_temp, y_temp = get_sequences(X[0], sequence_length, note_to_int, n_vocab)
    size_increment = (size_of_numpy_array(x_temp[0]) + size_of_numpy_array(y_temp[0])) / 1e6
    step = int(size_of_array / size_increment)
    x_temp = None
    y_temp = None


    # Build array of all sequences
    # Sequence a song
    song = X[i]
    x_, y_ = get_sequences(song, sequence_length, note_to_int, n_vocab)

    for j in range(0, len(x_)-step+1, step):
        # add the sequence to the x and y arrays
        X_seq.extend(x_[j:j+step])
        y.extend(y_[j:j+step])
        # save the array once it reaches the correct size
        with locker:
            # Save the file
            np.save(f'{save_path}/x{counter.value}', X_seq, allow_pickle=True)
            np.save(f'{save_path}/y{counter.value}', y, allow_pickle=True)
            # Reset memory
            X_seq.clear()
            y.clear()
            counter.value += 1

    print(f"Song {i+1}: complete")


def setup(train, val, sequence_length, note_to_int, n_vocab, size_of_array):
    # Sequence the songs and save to file
    counter_val_1 = Value('i', 0)
    lock_1 = Lock()

    num_processes = 6
    for j in range(0, len(val), num_processes):
        procs = [Process(target=threaded_sequencing,
                         args=(i, val, sequence_length, note_to_int, n_vocab, size_of_array, 'files/batch_data/val', counter_val_1, lock_1)) for i in
                 range(j, j + num_processes) if i < len(val)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        print('_' * 50)

    # choose training song
    # Sequence the songs and save to file
    counter_val = Value('i', 0)
    lock = Lock()

    for j in range(0, len(train), num_processes):
        procs = [Process(target=threaded_sequencing,
                         args=(i, train, sequence_length, note_to_int, n_vocab, size_of_array, 'files/batch_data/train', counter_val, lock)) for i in
                 range(j, j + num_processes) if i < len(train)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        print('_' * 50)


def getX_train_val():
    if glob.glob('files/X_train.npy') and glob.glob('files/X_val.npy') and get_user_yes_no(
            'Would you like to load the songs from memory'):
        X_train = np.load('files/X_train.npy', allow_pickle=True)
        X_val = np.load('files/X_val.npy', allow_pickle=True)
        pitchnames = read_pitchnames()
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        n_vocab = len(pitchnames)  # get amount of pitch names
    else:
        X, pitchnames, note_to_int, n_vocab = load_notes('files/midi_songs')
        X_train, X_val = train_test_split(X)
        if get_user_yes_no('Would you like to save the loaded data to memory'):
            np.save('files/X_train.npy', X_train, allow_pickle=True)
            np.save('files/X_val.npy', X_val, allow_pickle=True)
            save_pitchnames(pitchnames)
    return X_train, X_val, pitchnames, note_to_int, n_vocab


if __name__ == '__main__':
    freeze_support()

    sequence_length = 50
    size_of_array = 0.35
    X_train, X_val, pitchnames, note_to_int, n_vocab = getX_train_val()

    save_size_of_data((sequence_length, 1))

    # Clear old data
    for i in ['files/batch_data/train', 'files/batch_data/val']:
        delete_dir_contents(i)

    # Setup notes
    setup(X_train, X_val, sequence_length, note_to_int, n_vocab, size_of_array)
