import music21 as m21
import json

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
            nnames = "n({}, {})".format(
                # d.quarterLength,
                d.type,
                " ".join([p.nameWithOctave for p in n.pitches]))
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
    return out

if __name__ == '__main__':
    bcl = m21.corpus.chorales.ChoraleList()
    bcl.prepareList()
    with open('data.txt', 'w') as f:
        for num in bcl.byBWV.keys():
            identifier = "bach/bwv{}".format(num)
            c = m21.corpus.parse(identifier)
            score_txt = process_score(c)

            jsonl = json.dumps({
                'text': score_txt,
                'metadata': 'unused'
            }) + "\n"
            f.write(jsonl)


