from music21 import converter, instrument, note, chord
import glob
import numpy as np
from tqdm import tqdm
from keras import backend
import random
from collections.abc import Mapping, Iterable
from sys import getsizeof
from user_input import get_user_non_negative_number_or_default, get_user_yes_no


def randomize(x:list, n_iters=100):
    for i in range(n_iters):
        i1 = random.randrange(len(x))
        i2 = random.randrange(len(x))
        x[i1], x[i2] = x[i2], x[i1]
    return x


def deep_getsizeof(o, ids=None):
    """
    Find the memory footprint of a Python object

    This is a recursive function that drills down a Python object graph
    like a dictionary holding nested dictionaries with lists of lists
    and tuples and sets.

    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.

    :param o: the object
    :param ids:
    :return:
    """
    if ids is None:
        ids = set()

    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str) or isinstance(0, bytes):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())

    if isinstance(o, Iterable):
        return r + sum(d(x, ids) for x in o)

    return r


def load_song(song_path: str):
    midi = converter.parse(song_path)
    notes_to_parse = None
    parts = instrument.partitionByInstrument(midi)

    # Create 1 x 129 np array [0-127] note [128] duration
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
            # This gets me when the note is played
            # element.offset
            value = element.offset - lastOffset
            lastOffset = element.offset

            # set slice values
            if isinstance(element, note.Note):
                if not element.isRest:
                    # This gets me the note played in midi
                    # element.pitch.midi
                    thisSlice[element.pitch.midi] = 1
            elif isinstance(element, chord.Chord):
                for n in element.notes:
                    thisSlice[n.pitch.midi] = 1
            thisSlice[-1] = value
            notes.append(thisSlice)
            thisSlice = np.zeros(129, dtype=float)
    return notes


def load_notes(folder, num=None) -> list:
    notes = []
    # Change songs loaded later, took way too long on my desktop
    song_files = glob.glob(f"./{folder}/*.mid")
    for i, file in tqdm(enumerate(song_files), desc='Loading in songs'):
        if num is not None and i > num:
            return notes
        notes.append(np.array(load_song(file)))
    return notes


def get_sequences_durations(notes, sequence_length=70, data_multiplier=None, verbose=True) -> (np.array, np.array):
    X = []
    y = []

    # create input sequences and the corresponding outputs
    multiplier = 1
    if data_multiplier is not None:
        multiplier = sequence_length//data_multiplier
    for i in tqdm(range(0, int(len(notes) - sequence_length), int(multiplier)), desc='Segmenting songs into sequences', disable=(not verbose)):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        X.append(sequence_in)
        y.append(sequence_out[-1])

    return np.array(X), np.array(y)


def get_sequences_notes(notes, sequence_length=70, data_multiplier=None, verbose=True) -> (np.array, np.array):
    X = []
    y = []

    # create input sequences and the corresponding outputs
    multiplier = 1
    if data_multiplier is not None:
        multiplier = sequence_length//data_multiplier
    for i in tqdm(range(0, int(len(notes) - sequence_length), int(multiplier)), desc='Segmenting songs into sequences', disable=(not verbose)):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        X.append(sequence_in)
        y.append(sequence_out[:-1])

    return np.array(X), np.array(y)


def sequence_songs(X, sequence_length, sequence_method, data_multiplier=None, size_of_array=None, starting_position=0):
    """
    Builds X,Y arrays where X is a series of notes/chords and time steps and y is the target value
    :param X: Array of songs
    :param sequence_length: number of notes and chords used to predict the target y value
    :param sequence_method: 'duration' for time step target, 'notes' for note/chord target
    :param data_multiplier: The size of the output arrays compared to the input
    :param size_of_array: max size in MB of X and Y combined
    :param starting_position: place to begin indexing the song array
    :return: X array of n notes/chords, y target values
    """
    # Set sequence method based on string
    if sequence_method == 'duration':
        sequence_method = get_sequences_durations
    elif sequence_method == 'notes':
        sequence_method = get_sequences_notes
    else:
        raise ValueError(f'sequence method was "{sequence_method}", options are ["duration", "notes"]')

    X_seq = []
    y = []
    counter = starting_position
    current_size = 0

    if data_multiplier is None:
        data_multiplier = sequence_length

    while counter < len(X):
        # Sequence a song
        song = X[counter]
        counter += 1
        x_, y_ = sequence_method(song, sequence_length, data_multiplier=data_multiplier, verbose=False)

        # if capping result by size calculate size of arrays and break if the size would be too great
        if size_of_array is not None:
            current_size += deep_getsizeof(x_) + deep_getsizeof(y_)
            if (current_size/1e6) > size_of_array:
                break
            else:
                counter %= len(X)
        # add the sequence to the x and y arrays
        X_seq.extend(x_)
        y.extend(y_)

    X_seq = np.array(X_seq)
    y = np.array(y)
    return X_seq, y


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def r2_keras(y_true, y_pred):
    SS_res = backend.mean(backend.square(y_true - y_pred))
    SS_tot = backend.mean(backend.square(y_true - backend.mean(y_true)))
    return 1 - SS_res / (SS_tot + backend.epsilon())


def getX_train_val():
    if glob.glob('X_train.npy') and glob.glob('X_val.npy') and get_user_yes_no(
            'Would you like to load the songs from memory'):
        X_train = np.load('X_train.npy', allow_pickle=True)
        X_val = np.load('X_val.npy', allow_pickle=True)
    else:
        num_train = get_user_non_negative_number_or_default('How many training files do you want to load',
                                                            default_message='to load all files')
        X_train = load_notes('midi_songs/training', num_train)
        X_val = load_notes('midi_songs/testing')
        if get_user_yes_no('Would you like to save the loaded data to memory'):
            np.save('X_train.npy', X_train, allow_pickle=True)
            np.save('X_val.npy', X_val, allow_pickle=True)

    return X_train, X_val