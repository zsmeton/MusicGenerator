import matplotlib.pyplot as plt
from training_utils import load_song, get_sequences_durations
import glob

if __name__ == '__main__':
    for song_name in glob.glob('midi_songs/training/*')[:10]:
        song1 = load_song(song_name)
        song1_durations = [part[-1] for part in song1]
        print(song1_durations)
        plt.plot(song1_durations)
        plt.show()