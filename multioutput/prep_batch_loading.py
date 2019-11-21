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
            if lastOffset != element.offset:
                # This gets me when the note is played
                # element.offset
                value = element.offset - lastOffset
                lastOffset = element.offset
                notes.append(thisSlice)
                thisSlice = np.zeros(129, dtype=float)
            if isinstance(element, note.Note):
                if not element.isRest:
                    # This gets me the note played in midi
                    # element.pitch.midi
                    thisSlice[element.pitch.midi] = 1
            elif isinstance(element, chord.Chord):
                for n in element.notes:
                    thisSlice[n.pitch.midi] = 1
            thisSlice[-1] = value
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


def get_sequences_notes(notes, in_length, out_length, data_multiplier=None, verbose=False) -> (np.array, np.array):
    X = []
    y = []

    # create input sequences and the corresponding outputs
    multiplier = 1
    if data_multiplier is not None:
        multiplier = in_length // data_multiplier
    for i in tqdm(range(0, int(len(notes) - in_length - out_length - 1), int(multiplier)),
                  desc='Segmenting songs into sequences', disable=(not verbose)):
        sequence_in = notes[i:i + in_length]
        sequence_out = notes[i + in_length:i + in_length + out_length + 1]
        X.append(sequence_in)
        y.append([i[:-1] for i in sequence_out])

    return np.array(X), np.array(y)


def threaded_sequencing(i, X, in_length, out_length, size_of_array, save_path, counter: Value, locker: Lock):
    # Set sequence method based on string

    X_seq = []
    y = []
    current_size = 0

    # Chunk sequences into n sequences each of size of array
    x_temp, y_temp = get_sequences_notes(X[0], in_length, out_length, verbose=False)
    size_increment = (size_of_numpy_array(x_temp[0]) + size_of_numpy_array(y_temp[0])) / 1e6
    step = int(size_of_array / size_increment)
    x_temp = None
    y_temp = None

    # Build array of all sequences
    # Sequence a song
    song = X[i]
    x_, y_ = get_sequences_notes(song, in_length, out_length, verbose=False)

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


def setup(train, val, in_length, out_length, size_of_array):
    # Sequence the songs and save to file
    counter_val_1 = Value('i', 0)
    lock_1 = Lock()

    num_processes = 6
    for j in range(0, len(val), num_processes):
        procs = [Process(target=threaded_sequencing,
                         args=(i, val, in_length, out_length, size_of_array, f'batch_data/notes/val/', counter_val_1, lock_1)) for i in
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
        procs = [Process(target=threaded_sequencing, args=(i, train, in_length, out_length, size_of_array, f'batch_data/notes/train/', counter_val, lock)) for i in
                 range(j, j + num_processes) if i < len(train)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        print('_' * 50)


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


if __name__ == '__main__':
    freeze_support()

    in_sequence_length = 50
    out_sequence_length = 50
    size_of_array = 15
    X_train, X_val = getX_train_val()
    save_size_of_data((in_sequence_length, X_train[0].shape[1]))

    # Clear old data
    for i in ['batch_data/notes/train', 'batch_data/notes/val']:
        delete_dir_contents(i)

    # Setup notes
    setup(X_train, X_val, in_sequence_length, out_sequence_length, size_of_array)