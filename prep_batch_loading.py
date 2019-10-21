from ctypes import c_int

from training_utils import *
import os, shutil
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support, Value, Lock, Process


def save_size_of_data(X):
    with open('batch_data/shape.txt', 'w') as fout:
        fout.write(str(X[0].shape[1]))


def read_size_of_data():
    with open('batch_data/shape.txt', 'r') as fin:
        return int(fin.readline())


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


def threaded_sequencing(i, X, sequence_method_name, counter: Value, locker: Lock):
    # Set sequence method based on string
    if sequence_method_name == 'duration':
        sequence_method = get_sequences_durations
    elif sequence_method_name == 'notes':
        sequence_method = get_sequences_notes
    else:
        raise ValueError(f'sequence method was "{sequence_method_name}", options are ["duration", "notes"]')

    sequence_length = 70
    size_of_array = 50
    X_seq = []
    y = []
    current_size = 0

    # Chunk sequences into n sequences each of size of array
    x_temp, y_temp = sequence_method(X[0], sequence_length, verbose=False)
    size_increment = (size_of_numpy_array(x_temp[0]) + size_of_numpy_array(y_temp[0])) / 1e6
    x_temp = None
    y_temp = None

    # Build array of all sequences
    # Sequence a song
    song = X[i]
    x_, y_ = sequence_method(song, sequence_length, verbose=False)

    for x_elem, y_elem in zip(x_, y_):
        current_size += size_increment
        # add the sequence to the x and y arrays
        X_seq.append(x_elem)
        y.append(y_elem)
        # save the array once it reaches the correct size
        if current_size > size_of_array:
            with locker:
                # Save the file
                np.save(f'batch_data/{sequence_method_name}/train/x{counter.value}', X_seq, allow_pickle=True)
                np.save(f'batch_data/{sequence_method_name}/train/y{counter.value}', y, allow_pickle=True)
                # Reset memory
                X_seq.clear()
                y.clear()
                counter.value += 1
                current_size = 0

    if len(X_seq) != 0 and i+1 < len(X):
        song = X[i+1]
        x_, y_ = sequence_method(song, sequence_length, verbose=False)

        for x_elem, y_elem in zip(x_, y_):
            current_size += size_increment
            # add the sequence to the x and y arrays
            X_seq.append(x_elem)
            y.append(y_elem)
            # save the array once it reaches the correct size
            if current_size > size_of_array:
                with locker:
                    # Save the file
                    np.save(f'batch_data/{sequence_method_name}/train/x{counter.value}', X_seq, allow_pickle=True)
                    np.save(f'batch_data/{sequence_method_name}/train/y{counter.value}', y, allow_pickle=True)
                    # Reset memory
                    X_seq.clear()
                    y.clear()
                    counter.value += 1
                break

    print(f"Song {i}'s {sequence_method_name} data processing: complete")

"""
def sequence_songs_size(X, sequence_length, sequence_method_name, size_of_array):
    Builds X,Y arrays where X is a series of notes/chords and time steps and y is the target value
    :param X: Array of songs
    :param sequence_length: number of notes and chords used to predict the target y value
    :param sequence_method_name: 'duration' for time step target, 'notes' for note/chord target
    :param size_of_array: max size in MB of X and Y combined
    :return: X array of n notes/chords, y target values
    
    # Set sequence method based on string
    if sequence_method_name == 'duration':
        sequence_method = get_sequences_durations
    elif sequence_method_name == 'notes':
        sequence_method = get_sequences_notes
    else:
        raise ValueError(f'sequence method was "{sequence_method_name}", options are ["duration", "notes"]')

    X_seq = []
    y = []
    i=0
    current_size = 0
    counter = 0

    # Chunk sequences into n sequences each of size of array
    x_temp, y_temp = sequence_method(X[0], sequence_length, verbose=False)
    size_increment = (size_of_numpy_array(x_temp[0]) + size_of_numpy_array(y_temp[0])) / 1e6
    x_temp = None
    y_temp = None

    # Build array of all sequences
    pbar = tqdm(desc=f'Writing training data for {sequence_method_name} to files')
    while i < len(X):
        # Sequence a song
        song = X[i]
        i += 1
        x_, y_ = sequence_method(song, sequence_length, verbose=False)

        for x_elem, y_elem in zip(x_, y_):
            # save the array once it reaches the correct size
            current_size += size_increment
            if current_size > size_of_array:
                # Save the file
                np.save(f'batch_data/{sequence_method_name}/train/x{counter}', X_seq, allow_pickle=True)
                np.save(f'batch_data/{sequence_method_name}/train/y{counter}', y, allow_pickle=True)
                # Reset memory
                X_seq.clear()
                y.clear()
                counter += 1
                current_size = 0
                pbar.update(1)
            # add the sequence to the x and y arrays
            X_seq.append(x_elem)
            y.append(y_elem)
    pbar.close()
"""


def setup(train, val, method):
    # Setup duration
    X_val_sequence, y_val_sequence = sequence_songs(val, 70, sequence_method=method, data_multiplier=70 // 3)
    # Save the validation date to numpy array
    np.save(f'batch_data/{method}/val/x{0}', X_val_sequence, allow_pickle=True)
    np.save(f'batch_data/{method}/val/y{0}', y_val_sequence, allow_pickle=True)

    # choose training song
    # Sequence the songs and save to file
    counter_val = Value('i', 0)
    lock = Lock()

    for j in range(0, len(train), 4):
        procs = [Process(target=threaded_sequencing, args=(i, train, method, counter_val, lock)) for i in
                 range(j, j + 4) if i < len(train)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()


if __name__ == '__main__':
    freeze_support()

    X_train, X_val = getX_train_val()
    save_size_of_data(X_train)

    # Clear old data
    for i in ['batch_data/duration/train', 'batch_data/duration/val', 'batch_data/notes/train', 'batch_data/notes/val']:
        delete_dir_contents(i)

    # Setup duration
    setup(X_train, X_val, 'duration')

    # Setup notes
    setup(X_train, X_val, 'notes')
