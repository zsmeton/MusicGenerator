import random
import math
import numpy as np

class Generator:
    scale = [0, 2, 4, 5, 7, 9, 11] #Default scale is major
    offset =  60 #Default offset is middle C
                 #This leaves us with 4 octaves below, and 4 octaves above
    scale_degrees = len(scale)
    octave_range = 4
    p_free_note = 0.1
    quickest_note = 8
    beats = 4
    note_type = 4 #Default is 4/4 Time
    bpm = 90 #Default bpm is 90
    seed = 3078954
    p_dotted_note = 0.1
    p_chord = 0.4

    def _init_(self, scale=[0, 2, 4, 5, 7, 9, 11], offset=60, octave_range=4, p_free_note=0.1, quickest_note=8, beats=4, note_type=4, bpm=90, p_dotted_note=0.1, p_chord=0.4):
        self.scale = scale
        self.offset = offset
        self.octave_range = octave_range
        self.p_free_note = p_free_note
        self.scale_degrees = len(scale)
        self.quickest_note = quickest_note
        self.beats = beats
        self.note_type = note_type
        self.bpm = bpm
        self.p_dotted_note = p_dotted_note
        self.p_chord = p_chord

    '''Create and return an array with all of the time offsets to create 1 full measure of music'''
    def create_rhythm(self):
        seconds_per_beat = 1 / self.beats
        seconds_per_measure = seconds_per_beat * self.note_type

        beat = []

        while sum(beat) != seconds_per_measure:
            #Generate some number, translate it into a sensical note value
            duration = random.uniform(1, math.log(self.quickest_note, 2))
            duration = round(duration)

            #Translate note value into a time interval
            duration = (1 / (math.pow(2, duration)))
            if random.random() <= self.p_dotted_note:
                duration = 1.5 * duration

            #append to beat
            if sum(beat) + duration > seconds_per_measure:
                beat.append(seconds_per_measure - sum(beat))
            else:
                beat.append(duration)

        beat = [x / (((self.bpm / 60) / self.beats) * self.note_type) for x in beat]

        #Once beat has value seconds_per_measure, return it
        return beat

    def generate_notes(self):
        notes = []

        #Generate a single note
        new_note = math.floor(random.uniform(0, self.scale_degrees))
        #Transform it into an actual note
        new_note = self.scale[new_note] + self.offset
        octave_offset = random.random()

        if octave_offset < 0.05:
            new_note -= 24
        elif octave_offset < 0.2:
            new_note -= 12
        elif octave_offset > 0.95:
            new_note += 24
        elif octave_offset > 0.8:
            new_note += 12

        free_note_offset = random.random()

        if free_note_offset > self.p_free_note:
            new_note -= 1
        elif free_note_offset > self.p_free_note:
            new_note += 1

        notes.append(new_note)

        #Generate a chord if probability is met
        chord_probability = random.random()

        if chord_probability <= self.p_chord:
            #Generate a single note
            chord_base = math.floor(random.uniform(0, self.scale_degrees))
            #Transform it into an actual note
            chord_note = self.scale[chord_base] + self.offset - 12
            octave_offset = random.random()

            if octave_offset < 0.05:
                chord_note -= 24
            elif octave_offset < 0.2:
                chord_note -= 12
            elif octave_offset > 0.95:
                chord_note += 24
            elif octave_offset > 0.8:
                chord_note += 12

            notes.append(chord_note)

            p_next_note = 0.6

            while random.random() <= p_next_note:
                chord_base += 2
                note_offset = 0
                p_note_offset = random.random()
                if p_note_offset <= 0.05:
                    note_offset = -1
                elif p_note_offset >= 0.95:
                    note_offset = 1
                chord_note = self.scale[(chord_base + note_offset) % self.scale_degrees] + self.offset - 12

                octave_offset = random.random()

                if octave_offset < 0.05:
                    chord_note -= 24
                elif octave_offset < 0.2:
                    chord_note -= 12
                elif octave_offset > 0.95:
                    chord_note += 24
                elif octave_offset > 0.8:
                    chord_note += 12

                notes.append(chord_note)

                p_next_note = p_next_note * p_next_note

        #Return array with note indexes to be played
        return notes

    def new_offset(self, offset):
        self.offset = offset

    def generate_music(self, n_bars):
        sequence = None
        for i in range(0, n_bars):
            this_bar = self.create_rhythm()
            for j in this_bar:
                this_note = self.generate_notes()
                this_snapshot = np.zeros(129, dtype=float)
                for k in this_note:
                    this_snapshot[k] = 1
                this_snapshot[-1] = j

                #This is what I am talking about. The array is not appended to the end of the pressed keys sequence.
                if sequence is None:
                    sequence = np.array([this_snapshot])
                else:
                    sequence = np.append(sequence, np.array([this_snapshot]), axis=0)

        return sequence