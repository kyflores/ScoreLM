from multiprocessing import Pool
import json

import music21 as m21

# Turn music21 scores into strings. In short, the process is
# 1. Chordify the score. Chordify "flattens" multiple voices into chords in one voice.
# 2. Iterate through measures, turned every chord into a string of the form:
#    |n(0.5, G2 G3 B3 D4), n(0.5, G2 G3 A3 C4 D4)|
#    The format is |n(duration, note0, note1, note2, ...), n(...)|_|n(...)|
#    where noteN is a member of the chord. Measures are surrounded by |'s, and delimited by a _.
# 3. Write every score (yes the whoel thing) into a one line json, to produce a a jsonl dataset.

def stringize_measure(measure: m21.stream.Measure) -> str:
    # Possible infrequent data clef, tempo, key, meter,
    # Can contain Note, Chord
    out = []
    notes = measure.getElementsByClass(['Note', 'Chord']).stream().elements
    for n in notes:
        if isinstance(n, m21.chord.Chord):
            d = n.duration
            note_strings = []
            for p in n.notes:
                name = p.pitch.nameWithOctave
                tie = ""
                if p.tie is not None:
                    tietype = p.tie.type
                    if tietype == 'start' or tietype == 'continue':
                        tie = "T."
                note_strings.append("{}{}".format(tie, name))

            nnames = "n({}, {})".format(
                d.quarterLength,
                # d.type,
                " ".join(note_strings))
            out.append(nnames)
        elif isinstance(n, m21.note.Note):
            raise Exception("UNIMPLEMENTED")

    res = ", ".join(out)
    res = "|{}|".format(res)
    return res

def process_score(score):
    chords = score.chordify()
    els = chords.elements  # Tuple of stuff
    instrument = chords.getElementsByClass(m21.instrument.Instrument).first()
    meta = chords.getElementsByClass(m21.metadata.Metadata).first()
    measures = chords.measures(0, None).getElementsByClass(['Measure']).stream().elements
    measure_strings = [stringize_measure(m) for m in measures]
    out = "_".join(measure_strings)
    return out, None

def write_composer(composer: str):
    print('Serializing {}'.format(composer))
    pieces = m21.corpus.getComposer(composer)
    with open('dataset/{}.jsonl'.format(composer), 'w') as f:
        for p in pieces:
            print("Parsed {}".format(p))
            c = m21.corpus.parse(p)
            score_txt, meta = process_score(c)

            line = json.dumps({
                'text': score_txt,
                'metadata': 'unused'
            }) + "\n"
            f.write(line)


if __name__ == '__main__':
    bcl = m21.corpus.chorales.ChoraleList()
    bcl.prepareList()

    composers = [
        'bach',
        'beethoven',
        'monteverdi',
        'mozart',
        'palestrina',
        'ryansMammoth',
        'trecento',
    ]
    with Pool(8) as p:
        p.map(write_composer, composers)



