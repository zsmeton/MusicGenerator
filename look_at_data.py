import matplotlib.pyplot as plt
from training_utils import load_song, get_sequences_durations
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from music21 import converter, instrument, note, chord, pitch


def see_durations():
    for song_name in glob.glob('midi_songs/training/*')[:10]:
        song1 = load_song(song_name)
        song1_durations = [part[-1] for part in song1]
        plt.plot(song1_durations)
        plt.show()


def see_note_distribution():
    note_distribution = None
    X_train = np.load('X_train.npy', allow_pickle=True)
    for song in X_train:
        for part in song:
            if note_distribution is None:
                note_distribution = [int(i) for i in part[:-1]]
            else:
                for i, played in enumerate(part[:-1]):
                    if played > 0:
                        note_distribution[i] += 1
    x = np.arange(len(note_distribution))
    plt.bar(x, note_distribution)
    plt.title('Note Distribution')
    plt.savefig('visuals/note_distribution.png')
    plt.show()


def notes_per_timestep():
    timestep_distribution = dict()
    X_train = np.load('X_train.npy', allow_pickle=True)
    for song in X_train:
        for part in song:
            if sum(part[:-1]) in timestep_distribution:
                timestep_distribution[sum(part[:-1])] += 1
            else:
                timestep_distribution[sum(part[:-1])] = 1
    plt.bar(timestep_distribution.keys(), timestep_distribution.values())
    plt.savefig(f'visuals/note_timestep_distribution.png')
    plt.title("Distribution of notes per timestep")
    plt.show()


def notes_in_scale():
    key_distribution = dict()
    for song_name in tqdm(glob.glob('midi_songs/training/*'), desc='Analyzing pitch distribution based on song key'):
        midi = converter.parse(song_name)
        key = midi.analyze('key')
        dict_lookup = f"{key.tonic.name}-{key.mode}"
        if dict_lookup in key_distribution:
            curr_distribution = key_distribution[dict_lookup]
            song = load_song(song_name)
            for part in song:
                for i, played in enumerate(part[:-1]):
                    if played > 0:
                        curr_distribution[i] += 1
            key_distribution[dict_lookup] = curr_distribution

        else:
            song = load_song(song_name)
            if len(song) > 0:
                init_distribution = [int(i) for i in song[0][:-1]]
                for part in song[1:]:
                    for i, played in enumerate(part[:-1]):
                        if played > 0:
                            init_distribution[i] += 1
                key_distribution[dict_lookup] = init_distribution

    x = np.arange(128)
    c = pitch.Pitch('C4')
    ticks = []
    for i in x:
        c.midi = i
        ticks.append(c.name)

    for key, distribution in key_distribution.items():
        print(f'{key}:', distribution)
        plt.bar(x, distribution)
        plt.title(key)

        plt.xticks(x[::5], ticks[::5])
        plt.savefig(f'visuals/{key}_distribution.png')
        plt.show()


if __name__ == '__main__':
    #see_note_distribution()
    #notes_in_scale()
    notes_per_timestep()
