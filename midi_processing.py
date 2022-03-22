# this is an external library for processing midi files
import copy
import random
from music21 import *
import numpy as np
from collections import OrderedDict
from itertools import groupby, zip_longest

random.seed(4)


def scale_notes(_chord):
    scale_type = scale.MelodicMinorScale("F3")
    scales = scale_type.derive(_chord)
    _pitches = list(set([_pitch for _pitch in scales.getPitches()]))
    note_names = [i.name for i in _pitches]
    return note_names


def check_tone_in_scale(_chord, _note):
    return _note.name in scale_notes(_chord)


def check_tone_in_chord(_chord, _note):
    return _note.name in (_p.name for _p in _chord.pitches)


def chord_tone_generator(_chord):
    chord_name = [_p.nameWithOctave for _p in _chord.pitches]
    return note.Note(random.choice(chord_name))


def scale_tone_generator(_chord):
    note_names = scale_notes(_chord)
    _random_nome = random.choice(note_names)
    ordered_chord = _chord.sortAscending()
    random_octave = random.choice([i.octave for i in ordered_chord.pitches])
    scale_note = note.Note(("%s%s" % (_random_nome, random_octave)))
    return scale_note


def parser_stream(note_measure, chord_measure):
    note_measure_copy = copy.deepcopy(note_measure)
    chord_measure_copy = copy.deepcopy(chord_measure)
    note_measure_copy.removeByNotOfClass([note.Note, note.Rest])
    chord_measure_copy.removeByNotOfClass([chord.Chord])
    if len(note_measure_copy) == 0:
        return 0
    start_time = note_measure_copy[0].offset - (note_measure_copy[0].offset % 4)
    structure = ""
    previous_note = None
    i = 0
    for ix, nr in enumerate(note_measure_copy):
        try:
            N = [n for n in chord_measure_copy if n.offset <= nr.offset][-1]
        except IndexError:
            chord_measure_copy[0].offset = start_time
            N = [n for n in chord_measure_copy if n.offset <= nr.offset][-1]
        _nt = ' '
        if nr.name in N.pitchNames or isinstance(nr, chord.Chord):
            _nt = 'C'
        elif check_tone_in_scale(N, nr):
            _nt = 'S'
        else:
            _nt = 'C'
        note_info = "%s,%.3f" % (_nt, nr.quarterLength)
        interval_info = ""
        if isinstance(nr, note.Note):
            i += 1
            if i == 1:
                previous_note = nr
            else:
                P = interval.Interval(noteStart=previous_note, noteEnd=nr)
                U = interval.add([P, "m3"])
                f = interval.subtract([P, "m3"])
                interval_info = ",<%s,%s>" % (U.directedName, f.directedName)
                previous_note = nr
        musical_element = note_info + interval_info
        structure += (musical_element + " ")
    return structure.rstrip()


def un_parser(structure, _chords):
    stream_voice = stream.Voice()
    _offset = 0.0
    previous_element = None
    for ix, grammarElement in enumerate(structure.split(' ')):
        musical_element = grammarElement.split(',')
        _offset += float(musical_element[1])
        try:
            _chord = [n for n in _chords if n.offset <= _offset][-1]
        except IndexError:
            _chords[0].offset = 0.0
            _chord = [n for n in _chords if n.offset <= _offset][-1]
        if len(musical_element) == 2:
            if musical_element[0] == 'C':
                note_to_add = chord_tone_generator(_chord)
            else:
                note_to_add = scale_tone_generator(_chord)
            note_to_add.quarterLength = float(musical_element[1])
            if note_to_add.octave < 4:
                note_to_add.octave = 4
            stream_voice.insert(_offset, note_to_add)
            previous_element = note_to_add
        else:
            interval_1 = interval.Interval(musical_element[2].replace("<", ''))
            interval_2 = interval.Interval(musical_element[3].replace(">", ''))
            if interval_1.cents > interval_2.cents:
                _upper_interval, _lower_interval = interval_1, interval_2
            else:
                _upper_interval, _lower_interval = interval_2, interval_1
            _lower_pitch = interval.transposePitch(previous_element.pitch, _lower_interval)
            _higher_pitch = interval.transposePitch(previous_element.pitch, _upper_interval)
            _number_of_notes = int(_higher_pitch.ps - _lower_pitch.ps + 1)
            if musical_element[0] == 'C':
                u = []
                for i in range(0, _number_of_notes):
                    i = note.Note(_lower_pitch.transpose(i).simplifyEnharmonic())
                    if check_tone_in_chord(_chord, i):
                        u.append(i)
                if len(u) > 1:
                    note_to_add = random.choice([i for i in u if i.nameWithOctave != previous_element.nameWithOctave])
                elif len(u) == 1:
                    note_to_add = u[0]
                else:
                    note_to_add = previous_element.transpose(random.choice([-2, 2]))
                if note_to_add.octave < 3:
                    note_to_add.octave = 3
                note_to_add.quarterLength = float(musical_element[1])
                stream_voice.insert(_offset, note_to_add)
            elif musical_element[0] == 'S':
                k = []
                for i in range(0, _number_of_notes):
                    i = note.Note(_lower_pitch.transpose(i).simplifyEnharmonic())
                    if check_tone_in_scale(_chord, i):
                        k.append(i)
                if len(k) > 1:
                    note_to_add = random.choice([i for i in k if i.nameWithOctave != previous_element.nameWithOctave])
                elif len(k) == 1:
                    note_to_add = k[0]
                else:
                    note_to_add = previous_element.transpose(random.choice([-2, 2]))
                if note_to_add.octave < 3:
                    note_to_add.octave = 3
                note_to_add.quarterLength = float(musical_element[1])
                stream_voice.insert(_offset, note_to_add)
            else:
                M = []
                for i in range(0, _number_of_notes):
                    i = note.Note(_lower_pitch.transpose(i).simplifyEnharmonic())
                    M.append(i)
                if len(M) > 1:
                    note_to_add = random.choice([i for i in M if i.nameWithOctave != previous_element.nameWithOctave])
                elif len(M) == 1:
                    note_to_add = M[0]
                else:
                    note_to_add = previous_element.transpose(random.choice([-2, 2]))
                if note_to_add.octave < 3:
                    note_to_add.octave = 3
                note_to_add.quarterLength = float(musical_element[1])
                stream_voice.insert(_offset, note_to_add)
            previous_element = note_to_add
    return stream_voice


def _processing_(_corpus_, dictionary, n, op):
    number_of_elements = len(set(_corpus_))
    X = np.zeros((n, op, number_of_elements), dtype=np.bool)
    Y = np.zeros((n, op, number_of_elements), dtype=np.bool)
    for i in range(n):
        sequence_index = np.random.choice(len(_corpus_) - op)
        sequence = _corpus_[sequence_index:(sequence_index + op)]
        for j in range(op):
            index = dictionary[sequence[j]]
            if j != 0:
                X[i, j, index] = 1
                Y[i, j - 1, index] = 1
    Y = np.swapaxes(Y, 0, 1)
    Y = Y.tolist()
    return np.asarray(X), np.asarray(Y), number_of_elements


def __parser_midi(midi_file):
    midi_data = converter.parse(midi_file)
    _elements_of_midi_file = midi_data[1]
    stream_1, stream_2 = _elements_of_midi_file.getElementsByClass(stream.Voice)
    for j in stream_2:
        stream_1.insert(j.offset, j)
    _stream = stream_1
    for i in _stream:
        if i.quarterLength == 0.0:
            i.quarterLength = 0.25
    _stream.insert(0, instrument.Piano())
    _elements_of_midi_file = _stream
    _measures = OrderedDict()
    _offsets = [(int(n.offset / 4), n) for n in _elements_of_midi_file]
    number_of_measure = 0
    for _key, group in groupby(_offsets, lambda x: x[0]):
        _measures[number_of_measure] = [n[1] for n in group]
        number_of_measure += 1
    _stream.removeByClass(note.Rest)
    _stream.removeByClass(note.Note)
    _offset = [(int(n.offset / 4), n) for n in _stream]
    _chords = OrderedDict()
    number_of_measure = 0
    for _key, group in groupby(_offset, lambda x: x[0]):
        _chords[number_of_measure] = [n[1] for n in group]
        number_of_measure += 1
    return _measures, _chords


def getter_corpus(structure):
    _corpus_ = [x for sublist in structure for x in sublist.split(' ')]
    corpus_set = np.unique(np.array(_corpus_)).tolist()

    element_to_idx = dict()
    for i, element in enumerate(corpus_set):
        element_to_idx[element] = i

    idx_to_element = dict()
    for i, element in enumerate(corpus_set):
        idx_to_element[i] = element

    return _corpus_, element_to_idx, idx_to_element


def midi_to_data(midi_file):
    _measures, _chords = __parser_midi(midi_file)
    _structure = []
    for ix in range(1, len(_measures)):
        _stream = stream.Voice()
        for i in _measures[ix]:
            _stream.insert(i.offset, i)
        c = stream.Voice()
        try:
            for j in _chords[ix]:
                c.insert(j.offset, j)
            _parsed = parser_stream(_stream, c)
            if _parsed == 0:
                continue
            _structure.append(_parsed)
        except:
            None
    _corpus, dict_1, dict_2 = getter_corpus(_structure)
    X, Y, number = _processing_(_corpus, dict_1, 120, 40)
    return X, Y, number, _chords, dict_2, _corpus


def round_down(num, multiplier):
    return float(num) - (float(num) % multiplier)


def round_up(num, multiplier):
    return round_down(num, multiplier) + multiplier


def rounder(num, multiplier, up):
    if up < 0:
        return round_down(num, multiplier)
    else:
        return round_up(num, multiplier)


def __grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def prune_grammar(curr_grammar):
    pruned_grammar = curr_grammar.split(' ')
    for ix, gram in enumerate(pruned_grammar):
        terms = gram.split(',')
        terms[1] = str(rounder(float(terms[1]), 0.250,
                               random.choice([-1, 1])))
        pruned_grammar[ix] = ','.join(terms)
    pruned_grammar = ' '.join(pruned_grammar)
    return pruned_grammar



def data_to_midi(predictions, _chords, dictionary):
    out = stream.Stream()
    _distance = 0.0
    num_chords = int(len(_chords) / 3)
    for i in range(1, num_chords):
        chord_sequence = stream.Voice()
        for j in _chords[i]:
            chord_sequence.insert((j.offset % 4), j)
        predictions_list = predictions
        predictions_list = list(predictions_list.squeeze())
        predictions_translated = [dictionary[p] for p in predictions_list]
        predicted_tones = 'C,0.25 '
        for k in range(len(predictions_translated) - 1):
            predicted_tones += predictions_translated[k] + ' '
        predicted_tones += predictions_translated[-1]
        predicted_tones = prune_grammar(predicted_tones)
        sounds = un_parser(predicted_tones, chord_sequence)

        for m in sounds:
            out.insert(_distance + m.offset, m)
        for mc in chord_sequence:
            out.insert(_distance + mc.offset, mc)
        _distance += 4.0
    out.insert(0.0, tempo.MetronomeMark(number=80))
    return out
