import os
import re
import json
import random
import datetime
import numpy as np
import itertools as it

from keras.models import Sequential
from keras.layers import Dense, Dropout, Masking, LSTM, TimeDistributed


def subsets(iterable, minimum=0):
    for i in range(minimum, len(iterable) + 1):
        for x in it.combinations(iterable, i):
            yield x


def one_in_another(s1, s2):
    if s1 == s2:
        return True
    if len(s1) > len(s2):
        return one_in_another(s2, s1)
    for i in range(len(s2) + 1):
        for j in range(i):
            if s2.replace(s2[j:i], '') == s1:
                return True
    return False


class Chord(object):
    def __init__(self, root=None, intervals=set()):
        self.root = root
        self.intervals = intervals
        if root is not None:
            self.intervals.add(root)

    def __str__(self):
        return '(root: ' + str(self.root) + ' intervals: ' + ', '.join([str(x) for x in self.intervals]) + ')'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.root == other.root and self.intervals == other.intervals

    def transpose(self, half_steps=1):
        if self.root == None:
            return self
        return Chord((self.root + half_steps) % 12, set((i + half_steps) % 12 for i in self.intervals))

    def normalize(self):
        if self.root is None:
            return self
        return self.transpose(-1 * self.root)

    def add_note(self, note):
        return Chord(self.root, self.intervals.union([note % 12]))

    def signature(self):
        return [1 * (i == self.root) for i in range(12)] + [1 * (i in self.intervals) for i in range(12)]

    @staticmethod
    def all_chords():
        for x in subsets(range(12), minimum=3):
            temp = set(x)
            for i in temp:
                yield Chord(i, temp)


notes = [s + q for q in 'b#' for s in 'ABCDEFG'] + [s for s in 'ABCDEFG']


def isolate_stem(chord_word):
    if 'Bass' in chord_word:
        return 'Bass'
    if '\\' in chord_word:
        return
    elif '/' in chord_word:
        temp = chord_word.split('/')[0]
        for n in notes:
            if n in temp:
                return temp.replace(n, '')
    else:
        for n in notes:
            if n in chord_word:
                return chord_word.replace(n, '')


def stematize(chord):
    for note in notes:
        if note in chord:
            return stematize(chord.replace(note, str(note_to_num(note))))
    return chord


def note_to_num(note):
    notedict = {'A': 9, 'B': 11, 'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7}
    try:
        number = notedict[note[0]]
    except:
        number = 0
    if '#' in note:
        number = (number + 1) % 12
    if 'b' in note:
        number = (number - 1) % 12
    return number


note_cycle = dict()
for note in notes:
    temp = []
    n = note_to_num(note)
    for note2 in notes:
        if (note_to_num(note2) - 1) % 12 == n:
            temp += [note2]
    note_cycle[note] = temp


def stretch_chords(chords, length):
    c_len = len(chords)
    if c_len == 0:
        return []
    if c_len == length:
        return chords
    if length % c_len == 0:
        m = length // c_len
        out = []
        for chord in chords:
            out += [chord] * m
        return out
    return stretch_chords(chords + ['/'], length)


def process_leadsheets():
    sheet_dict = dict()
    for path in ['./the-imaginary-book-part-1-A-M/', './the-imaginary-book-part-2-N-Z/']:
        for fn in os.listdir(path=path):
            if '.ls' in fn:
                temp = []
                with open(path + fn) as file:
                    for line in file.readlines():
                        if 'meter' in line:
                            meter = line
                            meter = meter.replace(')', '').replace('(meter', '')
                            meter = meter.split()
                            meter = int(meter[0])
                        if '|' in line:
                            temp += re.sub("[\(\[].*?[\)\]]", "", line.replace('\n', '')).replace(')', '').split('|')
                temp = [x.split() for x in temp]
                temp = [x for x in temp if x != []]
                while True:
                    loop = False
                    for t in temp:
                        if len(t) > meter:
                            meter *= 2
                            loop = True
                            break
                    if not loop:
                        break
                temp = [stretch_chords(x, meter) for x in temp]
                temp = sum(temp, [])
                temp2 = []
                for i, t in enumerate(temp):
                    if t == '/':
                        temp2 += [temp2[i - 1]]
                    else:
                        temp2 += [t]
                temp = ' '.join(temp2)
                sheet_dict[fn.replace('.ls', '')] = temp
    with open('sheets.json', 'w') as file:
        json.dump(sheet_dict, file)


new_chord_defs = (
    'C.............. same as CM\nCM............. C major (c e g)\nC2............. same as CMadd9\nC5............. C five (c g)\nC6............. same as CM6\nC69............ same as CM69\nC6#11.......... same as CM6#11\nC69#11......... same as CM69#11\nC6b5........... same as CM6#11\nCM13........... C major thirteen (c e g b d a)\nCM13#11........ C major thirteen sharp eleven (c e g b d f# a)\nCmaj13......... same as CM13\nCMaj13......... same as CM13\nCmaj13#11...... same as CM13#11\nCMaj13#11...... same as CM13#11\nCM6............ C major six (c e g a)\nCM6#11......... C major six sharp eleven (c e g a f#)\nCM6b5.......... same as CM6#11\nCM69#11........ C major six nine sharp eleven (c e g a d f#)\nCM69........... C major six nine (c e g a d)\nCM7#11......... C major seven sharp eleven (c e g b f#)\nCM7............ C major seven (c e g b)\nCmaj7.......... same as CM7\nCMaj7.......... same as CM7\nCmaj7#11....... same as CM7#11\nCMaj7#11....... same as CM7#11\nCM7add13....... C major seven add 13 (c e g a b d)\nCM7b5.......... C major seven flat five (c e gb b)\nCM7b6.......... C major seven flat six (c e g ab b)\nCM7b9.......... C major seven flat nine (c e g b db)\nCM9............ C major nine (c e g b d)\nCM9#11......... C major nine sharp eleven (c e g b d f#)\nCmaj9.......... same as CM9\nCMaj9.......... same as CM9\nCmaj9#11....... same as CM9#11\nCMaj9#11....... same as CM9#11\nCM9b5.......... C major nine flat five (c e gb b d)\nCMadd9......... C major add nine (c e g d)\nCMb5........... C major flat five (c e gb)\nCMb6........... C major flat six (c e ab)\nCadd2.......... same as CMadd9\nCadd9.......... same as CMadd9\nCadd9no3....... same as CMsus2\n\nMinor Chords\n\nCm#5........... C minor sharp five (c eb g#)\nCm+............ same as Cm#5\nCm............. C minor (c eb g)\nCm11#5......... C minor eleven sharp five (c eb ab bb d f)\nCm11........... C minor eleven (c eb g bb d f)\nCm11b5......... C minor eleven flat five (c eb bb gb d f)\nCm13........... C minor thirteen (c eb g bb d f a)\nCm6............ C minor six (c eb g a)\nCm69........... C minor six nine (c eb g a d)\nCm7#5.......... C minor seven sharp five (c eb ab bb)\nCm7............ C minor seven (c eb g bb)\nCm7b5.......... C minor seven flat five (c eb gb bb)\nCh............. same as Cm7b5 (h for \"half-diminished\")\nCm9#5.......... C minor nine sharp five (c eb ab bb d)\nCm9............ C minor nine (c eb g bb d)\nCm9b5.......... C minor nine flat five (c eb bb gb d)\nCmM7........... C minor major seven (c eb g b)\nCmM7b6......... C minor major seven flat six (c eb g ab b)\nCmM9........... C minor major nine (c eb g b d)\nCmadd9......... C minor add nine (c eb g d)\nCmb6........... C minor flat six (c eb ab)\nCmb6M7......... C minor flat six major 7 (c eb ab b)\nCmb6b9......... C minor flat six flat nine (c eb ab db)\n\nDiminished Chords\n\nCdim........... C diminished triad (c eb gb)\nCo............. same as Cdim\nCdim7.......... C diminished seventh (c eb gb a)\nCo7............ same as Cdim7\nCoM7........... C diminished major seventh (c eb gb b)\nCo7M7.......... C diminished seventh major seventh (c eb gb a b)\n\nAugmented Chords\n\nCM#5........... C major sharp five (c e g#)\nC+............. same as CM#5\nCaug........... same as CM#5\nC+7............ same as C7#5\nCM#5add9....... C major sharp five add 9 (c e g# d)\nCM7#5.......... C major seven sharp five (c e g# b)\nCM7+........... same as CM7#5\nCM9#5.......... C major nine sharp five (c e g# b d)\nC+add9......... same as CM#5add9\n\nDominant Chords\n\nC7............. C seven (c e g bb)\nC7#5........... C seven sharp five (c e g# bb)\nC7+............ same as C7#5\nCaug7.......... same as C7#5\nC7aug.......... same as C7#5\nC7#5#9......... C seven sharp five sharp nine (c e g# bb d#)\nC7alt.......... same as C7#5#9\nC7b13.......... C seven flat thirteen (c e g bb ab)\nC7b5#9......... same as C7#9#11\nC7b5........... C seven flat five (c e gb bb)\nC7b5b13........ same as C7#11b13\nC7b5b9......... same as C7b9#11\nC7b5b9b13...... same as C7b9#11b13\nC7b6........... C seven flat six (c e g ab bb)\nC7b9#11........ C seven flat nine sharp eleven (c e g bb db f#)\nC7b9#11b13..... C seven flat nine sharp eleven flat thirteen (c e g bb db f# ab)\nC7b9........... C seven flat nine (c e g bb db)\nC7b9b13#11..... C seven flat nine flat thirteen sharp eleven (c e g bb db f# ab)\nC7b9b13........ C seven flat nine flat thirteen (c e g bb db ab)\nC7no5.......... C seven no five (c e bb)\nC7#11.......... C seven sharp eleven (c e g bb f#)\nC7#11b13....... C seven sharp eleven flat thirteen (c e g bb f# ab)\nC7#5b9#11...... C seven sharp five flat nine sharp 11 (c e g# bb db f#)\nC7#5b9......... C seven sharp five flat nine (c e g# bb db)\nC7#9#11........ C seven sharp nine sharp eleven (c e g bb d# f#)\nC7#9#11b13..... C seven sharp nine sharp eleven flat thirteen (c e g bb d# f# ab)\nC7#9........... C seven sharp nine (c e g bb d#)\nC7#9b13........ C seven sharp nine flat thirteen (c e g bb d# ab)\n\nC9............. C nine (c e g bb d)\nC9#5........... C nine sharp five (c e g# bb d)\nC9+............ same as C9#5\nC9#11.......... C nine sharp eleven (c e g bb d f#)\nC9#11b13....... C nine sharp eleven flat thirteen (c e g bb d f# ab)\nC9#5#11........ C nine sharp five sharp eleven (c e g# bb d f#)\nC9b13.......... C nine flat thirteen (c e g bb d ab)\nC9b5........... C nine flat five (c e gb bb d)\nC9b5b13........ same as C9#11b13\nC9no5.......... C nine no five (c e bb d)\n\nC13#11......... C thirteen sharp eleven (c e g bb d f# a)\nC13#9#11....... C thirteen sharp nine sharp eleven (c e g bb d# f# a)\nC13#9.......... C thirteen sharp nine (c e g bb d# a)\nC13............ C thirteen (c e g bb d a)\nC13b5.......... C thirteen flat five (c e gb a bb)\nC13b9#11....... C thirteen flat nine sharp eleven (c e g bb db f# a)\nC13b9.......... C thirteen flat nine (c e g bb db a)\n\nSuspensions\n\nCMsus2......... C major sus two (c d g)\nCMsus4......... C major sus four (c f g)\nCsus2.......... same as CMsus2\nCsus24......... C sus two four (c d f g)\nCsus4.......... same as CMsus4\nCsus4add9...... same as Csus24\nCsusb9......... C sus flat nine (c db f g)\nC4............. C four (c f bb eb)\nCquartal....... same as C4\nC7b9b13sus4.... same as C7sus4b9b13\nC7b9sus........ same as C7susb9\nC7b9sus4....... same as C7sus4b9\nC7b9sus4....... same as C7susb9\nC7sus.......... same as C7sus4\nC7sus4......... C seven sus four (c f g bb)\nC7sus4b9....... C seven sus four flat nine (c f g bb db)\nC7sus4b9b13.... C seven sus four flat nine flat thirteen (c f g bb db ab)\nC7susb9........ C seven sus flat nine (c db f g bb)\nC9sus4......... C nine sus four (c f g bb d)\nC9sus.......... same as C9sus4\nC11............ C eleven (c e g bb d f)\nC13sus......... same as C13sus4\nC13sus4........ C thirteen sus four (c f g bb d a)\n\nMiscellaneous\n\nCBlues......... C Blues (c eb f gb g bb) (Use upper case to avoid confusion with Cb = C flat)\nCBass.......... C Bass (c) (Use upper case to avoid confusion with Cb = C flat)\n')
chord_defs = [[y for y in x.split('.') if y != ''] for x in new_chord_defs.split('\n') if '.' in x]
equivs = [[x[0][1:], x[1].replace(' same as ', '').replace(' (h for "half-diminished")', '')[1:]] for x in chord_defs if
          ' same as ' in x[1]]
equivs += [['mMaj7', 'mM7'], ['h7', 'm7b5'], ['mb5', 'dim']]


def make_into_intervals(chord_def):
    temp = re.findall('\(([^\)]+)\)', chord_def[1])[0]
    temp = [chord_def[0][1:], [note_to_num(x.capitalize()) for x in temp.split()]]
    return [temp[0], [1 * (i in temp[1]) for i in range(12)]]


equiv_dict = dict([make_into_intervals(x) for x in chord_defs if ' same as ' not in x[1]])
equiv_dict = {**equiv_dict, **dict([[x[0], equiv_dict[x[1]]] for x in equivs])}
equiv_dict['NC'] = [0] * 12


def process_chord(chord):
    if 'NC' in chord:
        return Chord()
    if 'Bass' in chord:
        for note in notes:
            if chord[:len(note)] == note:
                break
        return Chord(note_to_num(note), {note_to_num(note)})
    if '\\' in chord:
        [poly, rest] = chord.split('\\')
        poly = process_chord(poly)
    else:
        poly = None
        rest = chord
    if '/' in chord:
        [shape, root] = rest.split('/')
    else:
        shape = rest
        root = None
    for note in notes:
        if shape[:len(note)] == note:
            break
    stem = shape.replace(note, '')
    chord_obj = Chord(0, set([i for i in range(12) if equiv_dict[stem][i]]))
    chord_obj = chord_obj.transpose(note_to_num(note))
    if root is not None:
        chord_obj.root = note_to_num(root)
    if poly is not None:
        for n in poly.intervals:
            chord_obj = chord_obj.add_note(n)
    return chord_obj


def process_leadsheet(sheet):
    return [process_chord(x) for x in sheet.split()]


def all_transpositions(sheet):
    temp = process_leadsheet(sheet)
    return [[chord.transpose(i) for chord in temp] for i in range(12)]


def transpose_note(note):
    return note_cycle[note][-1]


def transpose_chord_symbol(chord, interval=1):
    if interval == 0:
        return chord
    elif interval == 1:
        if 'NC' in chord:
            return 'NC'
        if 'Bass' in chord:
            for note in notes:
                if chord[:len(note)] == note:
                    break
            return transpose_note(note) + 'Bass'
        if '\\' in chord:
            return '\\'.join([transpose_chord_symbol(x) for x in chord.split('\\')])
        if '/' in chord:
            return '/'.join([transpose_chord_symbol(x) for x in chord.split('/')])
        for note in notes:
            if chord[:len(note)] == note:
                break
        stem = chord.replace(note, '')
        return transpose_note(note) + stem
    elif 1 < interval < 12:
        return transpose_chord_symbol(transpose_chord_symbol(chord, interval=interval - 1))
    else:
        return transpose_chord_symbol(chord, interval=(interval % 12))


def seq_generator(training_data, seq_len):
    i = 0
    l = len(training_data)
    while True:
        if i + seq_len + 1 > l:
            i = 0
        yield training_data[i:i + seq_len]  # , training_data[i + 1:i + seq_len + 1]
        i += 1


def batch_generator(seq_gen, batch_size):
    while True:
        X_out = []
        y_out = []
        for i in range(batch_size + 1):
            temp = next(seq_gen)
            if i != batch_size:
                X_out += [temp]
            if i != 0:
                y_out += [temp]
        yield np.array(X_out), np.array(y_out)


def strat_gen(data, batch_size, seq_len):
    data_len = len(data)
    data_slices = data_len // batch_size
    staggers = [data[i * data_slices:] + data[:i * data_slices] for i in range(batch_size)]
    staggers = [it.cycle(x) for x in staggers]
    X = [[next(x) for _ in range(seq_len)] for x in staggers]
    while True:
        y = [x[1:] + [next(staggers[i])] for i, x in enumerate(X)]
        yield np.array(X), np.array(y)
        X = [x for x in y]


def filename_string(dt):
    return '_{}_{}_{}_{}'.format(dt.toordinal(), dt.hour, dt.minute, dt.second)


def build_model(seq_len, batches, dropout_value=0.5, is_stateful=False, LSTM_size=512, LSTM_count=3):
    model = Sequential()

    model.add(Masking(batch_input_shape=(batches, seq_len, 24)))

    model.add(TimeDistributed(Dense(48, activation='relu')))
    model.add(Dropout(dropout_value))

    model.add(TimeDistributed(Dense(48, activation='relu')))
    model.add(Dropout(dropout_value))

    for _ in range(LSTM_count):
        model.add(LSTM(
            LSTM_size,
            return_sequences=True,
            recurrent_dropout=dropout_value,
            dropout=dropout_value,
            stateful=is_stateful,
            implementation=2
        ))

    model.add(TimeDistributed(Dense(24, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    with open('sheets.json') as file:
        sheet_dict = json.load(file)
    print('Sheets loaded')

    every_sheet = []
    for sheet in sheet_dict.values():
        every_sheet += all_transpositions(sheet)
    print('Transpositions applied')

    training_data = [[x.signature() for x in sheet] for sheet in every_sheet]
    print('Training data processed')

    random.shuffle(training_data)

    out = []
    for x in training_data:
        out += x
        out += [24 * [0]] * 8
    print('Training data ready for usage')

    last_epoch = 0
    epoch_stride = 1

    seq_len = 32
    is_stateful = False
    dropout_value = 0.5
    LSTM_size = 1024
    LSTM_count = 3
    if is_stateful:
        batches = 1
    else:
        batches = 4 * seq_len

    model = build_model(
        seq_len=seq_len,
        batches=batches,
        dropout_value=dropout_value,
        is_stateful=is_stateful,
        LSTM_size=LSTM_size,
        LSTM_count=LSTM_count
    )

    print(model.summary())

    out = np.asarray(out)
    gen = batch_generator(seq_generator(out, seq_len), batches)

    histories = []

    for _ in range(10):
        history = model.fit_generator(
            gen,
            steps_per_epoch=(len(out) // batches),
            epochs=(last_epoch + epoch_stride),
            verbose=1,
            initial_epoch=last_epoch
        )

        histories += [history]

        last_epoch += epoch_stride

        model_name = 'model_'
        if is_stateful:
            model_name += 'is_stateful_'
        else:
            model_name += 'not_stateful_'

        right_now = datetime.datetime.now()

        model_name += str(last_epoch)
        model_name += '_epochs'
        model_name += filename_string(right_now)

        model.save(model_name)
