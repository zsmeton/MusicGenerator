from keras import Sequential

from training_utils import *
from prep_batch_loading import read_size_of_data
import user_input
import train_notes
import train_duration
from music21 import pitch, stream


def get_durations(song):
    return [slice[-1] for slice in song]


def get_notes(song):
    return [slice[:-1] for slice in song]


def round_cutoff(num, cutoff = 0.5):
    """
    Turns a number into 1 >= cutoff or 0 if < cutoff
    :param num: number to 'round'
    :param cutoff: the cutoff point
    :return: 0 or 1
    """
    if num >= cutoff:
        return 1.0
    else:
        return 0.0

def generate_music(duration_model: Sequential, notes_model: Sequential, seed_song, song_length, savefile):
    # Run on some starter data
    notes_model.predict(seed_song)
    duration_model.predict(seed_song)

    # generate 500 notes
    song_durations = get_durations(seed_song[0])
    song_notes = get_notes([list(slice) for slice in seed_song[0]])
    pattern = seed_song
    pb = tqdm(desc='Generate music')
    while sum(song_durations) < song_length:
        # Predict next slice and append to song notes and duration
        predicted_notes = notes_model.predict(pattern)
        print(max(list(predicted_notes[0])))
        predicted_notes = list(np.apply_along_axis(round_cutoff, 0, predicted_notes))
        song_notes.append(predicted_notes)
        predicted_duration = duration_model.predict(pattern)[0][0].item()
        song_durations.append(float(predicted_duration))
        # update the seed for the next sequence to be last sequence_length - 1 + newest predcitions
        pattern = list(pattern[0][1:])
        predicted_notes.append(predicted_duration)
        pattern.append(np.array(predicted_notes))
        pattern = np.array([np.array(pattern)])
        pb.update(1)

    offset = 0
    output_notes = []
    # create note and chord objects based on the values generated by the model
    for notes, duration in zip(song_notes, song_durations):
        if sum(notes) > 0:
            notes_in_chord = [int(index[0]) for index in np.array(notes).nonzero()]
        else:
            notes_in_chord = []
        chord_notes = []
        for current_note in notes_in_chord:
            pitch_temp = pitch.Pitch()
            pitch_temp.midi = current_note
            new_note = note.Note(pitch_temp.name)
            new_note.storedInstrument = instrument.Piano()
            chord_notes.append(new_note)
        # notes are chord
        if len(chord_notes) > 1:
            curr_chord = chord.Chord(chord_notes)
            curr_chord.offset = offset
            output_notes.append(curr_chord)
        # notes are note
        elif len(chord_notes) == 1:
            curr_note = chord_notes[0]
            curr_note.offset = offset
            output_notes.append(curr_note)
        # note is rest
        else:
            curr_note = note.Note()
            curr_note.isRest = True
        # increase offset each iteration so that notes do not stack
        offset += duration

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=savefile + '.mid')


if __name__ == "__main__":
    option = user_input.get_user_options('What would you like to do', ['Generate music', 'Exit'])
    while option < 2:
        # get the song and model files
        seed_song_file = 'midi_songs/training/bach_846.mid' # user_input.get_user_filename('Path to seed song', '.mid')
        duration_model_file = 'models/duration/duration-model-20-0.0936.hdf5' # user_input.get_user_filename('Path to duration model', '.hdf5')
        notes_model_file = 'models/notes/note-model-05-0.0395.hdf5' # user_input.get_user_filename('Path to notes model', '.hdf5')
        save_filename = 'songs/song1'  # user_input.get_user_filename('Path to notes model', '.hdf5')
        # get length of song to generate
        length = 120 # int(user_input.get_user_non_negative_number('How long should the song be in seconds'))
        while length <= 0:
            print('Cannot generate song of zero time')
            length = int(user_input.get_user_non_negative_number('How long should the song be in seconds'))

        # load in the data
        sequence_length = read_size_of_data()[0]

        duration_model = train_duration.create_model_duration(read_size_of_data())
        duration_model.load_weights(duration_model_file)

        notes_model = train_notes.create_model_notes(read_size_of_data())
        notes_model.load_weights(notes_model_file)

        seed_song = np.array([load_song(seed_song_file)[50:100]])

        # generate music
        generate_music(duration_model, notes_model, seed_song, length, save_filename)

        option = user_input.get_user_options('What would you like to do', ['Generate music', 'Exit'])