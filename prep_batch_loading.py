from ctypes import c_int

from training_utils import *
import os, shutil
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support, Value, Lock, Process

def save_size_of_data(size):
    with open('batch_data/shape.txt', 'w') as fout:
        fout.write(str(size))


def read_size_of_data():
    with open('batch_data/shape.txt', 'r') as fin:
        return eval(fin.readline())


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


def threaded_sequencing(i, X, sequence_length, size_of_array, sequence_method_name, save_path, counter: Value, locker: Lock):
    # Set sequence method based on string
    if sequence_method_name == 'duration':
        sequence_method = get_sequences_durations
    elif sequence_method_name == 'notes':
        sequence_method = get_sequences_notes
    else:
        raise ValueError(f'sequence method was "{sequence_method_name}", options are ["duration", "notes"]')

    X_seq = []
    y = []
    current_size = 0

    # Chunk sequences into n sequences each of size of array
    x_temp, y_temp = sequence_method(X[0], sequence_length, verbose=False)
    size_increment = (size_of_numpy_array(x_temp[0]) + size_of_numpy_array(y_temp[0])) / 1e6
    step = int(size_of_array / size_increment)
    x_temp = None
    y_temp = None


    # Build array of all sequences
    # Sequence a song
    song = X[i]
    x_, y_ = sequence_method(song, sequence_length, verbose=False)

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

    print(f"Song {i+1}'s {sequence_method_name} data processing: complete")


def setup(train, val, method, sequence_length, size_of_array):
    # Sequence the songs and save to file
    counter_val_1 = Value('i', 0)
    lock_1 = Lock()

    num_processes = 6
    for j in range(0, len(val), num_processes):
        procs = [Process(target=threaded_sequencing,
                         args=(i, val, sequence_length, size_of_array, method, f'batch_data/{method}/val/', counter_val_1, lock_1)) for i in
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
        procs = [Process(target=threaded_sequencing, args=(i, train, sequence_length, size_of_array, method, f'batch_data/{method}/train/', counter_val, lock)) for i in
                 range(j, j + num_processes) if i < len(train)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        print('_' * 50)


def note_lookup_dict(filename):
    """
    Create a lookup table of integers to pitch for a song
    ex. 60 = 'C4'
    :param filename: path to song
    :return:
    """


if __name__ == '__main__':
    freeze_support()

    sequence_length = 75
    size_of_array = 15
    X_train, X_val = getX_train_val()
    save_size_of_data((sequence_length, X_train[0].shape[1]))


    # Clear old data
    for i in ['batch_data/duration/train', 'batch_data/duration/val', 'batch_data/notes/train', 'batch_data/notes/val']:
        delete_dir_contents(i)

    # Setup duration
    setup(X_train, X_val, 'duration', sequence_length, size_of_array)

    # Setup notes
    setup(X_train, X_val, 'notes', sequence_length, size_of_array)