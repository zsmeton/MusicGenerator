import music21
from tqdm import tqdm
from src.prep_batch_loading import read_pitchnames, load_notes, load_song, read_size_of_data
import numpy as np
from music21 import converter, instrument, note, chord
from src.train_notes import create_model


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_music(l_model, input_file, output_file, temperature=1.0):
    # Get model information
    pitchnames = read_pitchnames()
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    n_vocab = len(pitchnames)  # get amount of pitch names
    sequence_length = read_size_of_data()[0]

    # Get seed music from file
    seed = load_song(input_file)
    # TODO: AVERAGE NOTE DURATION OF SONG
    if not seed:
        raise ValueError("Empty song")
    # Check if we can run on the song
    cannot_parse = any([True for item in seed if item not in pitchnames])
    if cannot_parse:
        raise ValueError("Song contains notes or chords that is model was not trained to handle")

    # Format song for network
    network_seed = []
    # Pad array with zeros
    for i in range(sequence_length - len(seed)):
        network_seed.append(0)
    # Add notes
    for i in range(max(len(seed) - sequence_length, 0), len(seed)):
        network_seed.append(note_to_int[seed[i]])

    # Generate notes
    pattern = network_seed
    # TODO: Return back to appending generated song ontop of full song
    prediction_output = []
    # TODO: set for loop range based on desired duration
    for note_index in tqdm(range(300), desc='Generating music'):
        prediction_input = np.reshape(pattern, (1, sequence_length))
        prediction = l_model.predict(prediction_input, verbose=0)
        prediction = prediction.flatten()
        index = sample(prediction, temperature)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:sequence_length + 1]

    offset = 0
    output_notes = []
    # create note and chord objects based on the values generated by the model
    for section in prediction_output:
        # section is a chord
        if ('.' in section) or section.isdigit():
            notes_in_chord = section.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = music21.note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # section is a note
        else:
            new_note = music21.note.Note(section)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = music21.stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)


if __name__ == "__main__":
    # load in model
    n_vocab = len(read_pitchnames())  # get amount of pitch names
    model = create_model(read_size_of_data(), n_vocab)
    model.summary()
    model.load_weights("../files/models/notes/model-12-4.7494.hdf5")

    # Generate music
    generate_music(model, "../files/midi_songs/bach_846.mid", "../files/songs/bach_1.mid")
