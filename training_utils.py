from music21 import converter, instrument, note, chord, stream
import glob
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from plot_losses import PlotLearning
from user_input import get_user_yes_no, get_user_options, get_user_non_negative_number, get_user_filename, get_user_non_negative_number_or_default
from tqdm import tqdm
from guppy import hpy
from keras import backend
import random
from sklearn.metrics.regression import r2_score


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
    """

    :param notes:
    :param sequence_length:
    :param data_multiplier:
    :param verbose:
    :return:
    """
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


def sequence_songs(X, sequence_length, data_multiplier=None):
    X_seq = []
    y = []
    for song in X:
        x_, y_ = get_sequences_durations(song, sequence_length, data_multiplier=data_multiplier, verbose=False)
        X_seq.extend(x_)
        y.extend(y_)
    X_seq = np.array(X_seq)
    y = np.array(y)
    return X_seq, y


def sequence_songs_size(X, starting_position, size_of_array, sequence_length, data_multiplier=None):
    X_seq = []
    y = []
    counter = starting_position
    while len(X_seq) <= size_of_array:
        song = X[counter]
        x_, y_ = get_sequences_durations(song, sequence_length, data_multiplier=data_multiplier, verbose=False)
        X_seq.extend(x_)
        y.extend(y_)
        counter += 1
    X_seq = np.array(X_seq)
    y = np.array(y)
    return X_seq, y


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def r2_keras(y_true, y_pred):
    SS_res = backend.mean(backend.square(y_true - y_pred))
    SS_tot = backend.mean(backend.square(y_true - backend.mean(y_true)))
    return 1 - SS_res / (SS_tot + backend.epsilon())