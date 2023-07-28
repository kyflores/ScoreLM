import json
import sys
import re

import music21 as m21
import transformers as tfs
import torch

teststr = "|n(quarter, F3 A3 C4 F4)|_|n(eighth, E3 G3 C4 G4 C5), n(eighth, E3 G3 C4 C5), n(eighth, F3 A3 C4 F4 A4), n(eighth, F3 A3 C4 A4), n(eighth, D3 A3 D4 F4), n(eighth, D3 F3 A3 D4 F4), n(eighth, A2 A3 F4 C5), n(eighth, A2 A3 C4 F4 C5)|_|n(16th, B-2 F3 F4 B-4 D5), n(16th, B-2 F3 B-4 D5), n(16th, C3 F3 A4 B-4 D5), n(16th, C3 F3 G4 B-4 D5), n(eighth, D3 F4 B-4 D5), n(eighth, E3 F4 G4 B-4 D5), n(quarter, F3 F4 A4 C5), n(quarter, F3 F4 A4 C5)|_|n(eighth, B-3 F4 B-4 D5), n(16th, B-3 F4 G4 B-4 D5), n(16th, B-3 F4 A4 B-4 D5), n(eighth, A3 G4 B-4 E5), n(16th, G3 G4 B-4 C5 E5), n(16th, G3 G4 B-4 E5), n(eighth, A3 F4 A4 C5 F5), n(eighth, B3 D4 G4 D5 F5), n(eighth, C4 E4 G4 C5 E5), n(eighth, C4 E4 G4 C5 E5)|_|n(eighth, F3 F4 A4 C5 D5), n(16th, F3 E4 A4 B4 D5), n(16th, F3 E4 A4 D5), n(quarter, G3 D4 G4 B4 D5), n(quarter, C3 E4 G4 C5), n(eighth, F3 C4 F4 A4), n(eighth, F3 A3 F4 A4)|_|n(eighth, B-2 F3 F4 D5), n(eighth, B-2 G3 F4 B-4 D5), n(eighth, C3 A3 E4 A4 C5), n(eighth, C3 A3 E4 G4 C5), n(eighth, D3 A3 F4 B-4), n(eighth, E3 G3 C4 G4 B-4), n(eighth, F3 A3 C4 F4 A4), n(eighth, F3 C4 F4 A4)|_|n(eighth, C3 G3 C4 F4 G4), n(16th, C3 A3 F4 G4), n(16th, C3 F3 A3 F4 G4), n(eighth, C3 B-3 C4 E4 G4), n(16th, C3 B-3 C4 E4 G4), n(16th, C3 B-3 C4 E4 G4), n(quarter, F2 A3 C4 F4), n(quarter, F3 A3 C4 F4)|_|n(eighth, E3 G3 C4 G4 C5), n(eighth, E3 G3 C4 C5), n(eighth, F3 A3 C4 F4 A4), n(eighth, F3 A3 C4 A4), n(eighth, D3 A3 D4 F4), n(eighth, D3 F3 A3 D4 F4), n(eighth, A2 A3 F4 C5), n(eighth, A2 A3 C4 F4 C5)|_|n(16th, B-2 F3 F4 B-4 D5), n(16th, B-2 F3 B-4 D5), n(16th, C3 F3 A4 B-4 D5), n(16th, C3 F3 G4 B-4 D5), n(eighth, D3 F4 B-4 D5), n(eighth, E3 F4 G4 B-4 D5), n(quarter, F3 F4 A4 C5), n(quarter, F3 F4 A4 C5)|_|n(eighth, B-3 F4 B-4 D5), n(16th, B-3 F4 G4 B-4 D5), n(16th, B-3 F4 A4 B-4 D5), n(eighth, A3 G4 B-4 E5), n(16th, G3 G4 B-4 C5 E5), n(16th, G3 G4 B-4 E5), n(eighth, A3 F4 A4 C5 F5), n(eighth, B3 D4 G4 D5 F5), n(eighth, C4 E4 G4 C5 E5), n(eighth, C4 E4 G4 C5 E5)|_|n(eighth, F3 F4 A4 C5 D5), n(16th, F3 E4 A4 B4 D5), n(16th, F3 E4 A4 D5), n(quarter, G3 D4 G4 B4 D5), n(quarter, C3 E4 G4 C5), n(eighth, F3 C4 F4 A4), n(eighth, F3 A3 F4 A4)|_|n(eighth, B-2 F3 F4 D5), n(eighth, B-2 G3 F4 B-4 D5), n(eighth, C3 A3 E4 A4 C5), n(eighth, C3 A3 E4 G4 C5), n(eighth, D3 A3 F4 B-4), n(eighth, E3 G3 C4 G4 B-4), n(eighth, F3 A3 C4 F4 A4), n(eighth, F3 C4 F4 A4)|_|n(eighth, C3 G3 C4 F4 G4), n(16th, C3 A3 F4 G4), n(16th, C3 F3 A3 F4 G4), n(eighth, C3 B-3 C4 E4 G4), n(16th, C3 B-3 C4 E4 G4), n(16th, C3 B-3 C4 E4 G4), n(quarter, F2 A3 C4 F4)|_|n(eighth, E3 C4 G4 C5), n(16th, E3 C4 G4 C5), n(16th, E3 C4 G4 C5), n(16th, E3 C4 G4 C5), n(16th, E3 C4 G4 C5), n(16th, E3 C4 G4 C5), n(16th, E3 C4 G4 C5), n(half, F3 C4 F4 A4)|_|n(16th, C3 C4 F4 G4 C5), n(16th, C3 C4 F4 C5), n(16th, C3 C4 E4 G4 C5), n(16th, C3 C4 D4 A4 C5), n(quarter, C3 C4 E4 G4 C5), n(half, F3 C4 F4 A4)|_|n(eighth, F3 C4 F4 A4), n(eighth, G3 C4 F4 A4), n(eighth, A3 C4 F4 A4), n(eighth, F3 C4 F4 A4), n(eighth, C3 C4 E4 G4 C5), n(16th, D3 C4 E4 G4 B-4), n(16th, D3 C4 E4 G4 A4), n(eighth, E3 C4 E4 G4 B-4), n(16th, C3 C4 E4 G4 C5), n(16th, C3 C4 E4 G4 B-4)|_|n(eighth, F3 C4 F4 A4), n(eighth, G3 C4 F4 G4 A4), n(eighth, A3 C4 F4 A4), n(eighth, F3 C4 F4 A4), n(quarter, C4 E4 G4 C5), n(eighth, B-3 C4 F4 A4 C5), n(eighth, A3 C4 F4 A4)|_|n(eighth, G3 D4 G4 B-4), n(16th, G3 D4 F4 G4 B-4), n(16th, G3 D4 E4 G4 B-4), n(eighth, D4 F4 A4), n(eighth, C4 D4 E4 F4 A4), n(eighth, B-3 D4 F4 G4), n(eighth, F3 B-3 D4 G4), n(eighth, C4 E4 G4), n(16th, C4 E4 G4), n(16th, C4 E4 G4)|_|n(half, F3 A3 C4 F4), n(16th, D3 D4 A4 F5), n(16th, D3 D4 G4 A4 F5), n(16th, F3 D4 A4 F5), n(16th, F3 D4 A4 B-4 F5), n(eighth, A3 C4 A4 C5 E5), n(eighth, A3 C4 G4 C5 E5)|_|n(eighth, B-2 D4 F4 C5 D5), n(eighth, D3 B-3 F4 B-4 D5), n(eighth, F3 F4 B-4 C5), n(eighth, F3 A3 F4 A4 C5), n(eighth, G3 B-3 F4 A4 B-4), n(eighth, G3 B-3 E4 G4 B-4), n(eighth, A3 C4 G4 A4), n(eighth, D3 A3 F4 A4)|_|n(eighth, B2 D4 F4 G4), n(eighth, B2 F3 D4 F4 G4), n(16th, C3 G3 C4 E4 G4), n(16th, C3 F3 C4 E4 G4), n(16th, C3 G3 C4 E4 G4), n(16th, C3 G3 C4 E4 G4), n(quarter, F2 A3 C4 F4)|"
teststr = "|n(0.5, F#3 A3 C#4 F#4), n(0.5, F#3 B3 C#4 G#4)|_|n(0.25, F#3 C#4 F#4 A4), n(0.25, G#3 C#4 F#4 A4), n(0.25, A3 C#4 F#4 A4), n(0.25, B3 C#4 F#4 A4), n(0.5, C#4 E#4 G#4), n(0.5, C#3 B3 E#4 G#4), n(1.0, D3 A3 F#4), n(0.5, D3 B3 F#4), n(0.5, D3 B3 F#4 G#4)|_|n(0.5, C#3 C#4 F#4 A4), n(0.5, B2 C#4 F#4 A4), n(0.5, C#3 C#4 E#4 G#4), n(0.5, C#3 B3 E#4 G#4), n(1.0, F#2 A3 C#4 F#4), n(1.0, F#3 A3 F#4 C#5)|_|n(0.5, G#3 B3 F#4 B4), n(0.5, G#3 B3 E#4 B4), n(0.5, A3 C#4 F#4 A4), n(0.5, B3 C#4 F#4 A4), n(1.0, C#4 E#4 G#4), n(1.0, C#3 C#4 E#4 G#4)|_|n(0.5, F#3 C#4 F#4 A4), n(0.5, E3 C#4 F#4 A4), n(0.5, D3 D4 F#4 A4), n(0.5, C#3 D4 F#4 A4), n(0.5, D3 D4 F#4 B4), n(0.25, B2 E4 G#4 B4), n(0.25, B2 F#4 A4 B4), n(0.5, E3 E4 G#4 B4), n(0.5, E2 E3 D4 G#4 B4)|_|n(0.5, A2 C#4 G#4 C#5), n(0.5, A3 C#4 F#4 C#5), n(0.5, G#3 B3 E#4 C#5), n(0.5, F#3 A3 F#4 C#5), n(0.5, F#3 D4 G#4 B4), n(0.5, E#3 C#4 G#4 B4), n(0.5, F#3 C#4 G#4 A4), n(0.5, D3 D4 F#4 A4)|_|n(0.5, B2 D4 F#4 G#4), n(0.5, G#2 B3 F#4 G#4), n(0.5, C#3 G#3 E#4 G#4), n(0.5, C#3 C#4 E#4 G#4), n(1.0, F#2 A3 C#4 F#4), n(0.5, F#3 A3 F#4 C#5), n(0.5, E3 A3 F#4 C#5)|_|n(0.5, D3 B3 F#4 B4), n(0.5, C#3 C#4 E#4 B4), n(0.5, B#2 D#4 F#4 A4), n(0.5, B#2 D#4 F#4 G#4), n(1.0, C#3 C#4 E#4 G#4), n(0.5, A2 C#4 F#4 C#5), n(0.5, A2 C#4 E4 C#5)|_|n(0.5, B2 F#3 D#4 B4), n(0.5, C#3 F#3 E4 B4), n(0.5, D#3 B3 F#4 A4), n(0.5, B2 B3 D#4 A4), n(1.0, E3 B3 E4 G#4), n(0.5, E#3 C#4 G#4), n(0.5, C#3 C#4 E#4 G#4)|_|n(0.5, F#3 C#4 F#4 A4), n(0.5, F#2 F#3 A3 F#4 A4), n(0.5, F#3 D4 A4), n(0.5, D3 D4 F#4 A4), n(0.5, G3 D4 B4), n(0.5, G2 G3 B3 D4 B4), n(0.5, G#3 E4 B4), n(0.5, E3 E4 G#4 B4)|_|n(0.5, A3 E4 A4 C#5), n(0.5, A2 A3 C#4 G4 C#5), n(0.5, A#3 F#4 C#5), n(0.5, F#3 E4 F#4 C#5), n(0.5, B3 D4 F#4 B4), n(0.5, B2 B3 C#4 E#4 B4), n(0.5, B#3 D#4 F#4 A4), n(0.5, G#3 D#4 F#4 G#4)|_|n(0.5, C#4 F#4 G#4), n(0.25, G#3 B3 C#4 E#4 G#4), n(0.25, G#3 B3 C#4 D#4 G#4), n(0.5, C#3 C#4 E#4 G#4), n(0.5, C#3 B3 E#4 G#4), n(1.0, F#2 F#3 A3 C#4 F#4)|"

def infer(cfg):
    tokenizer = tfs.AutoTokenizer.from_pretrained(
        cfg['model_name'],
        model_max_length=cfg['blocksize'],
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = tfs.AutoModelForCausalLM.from_pretrained(
        "score-lm",
    ).to(cfg['device'])

    start = None
    if len(sys.argv) == 1:
        start = "|"
    else:
        start = sys.argv[1]

    inputs = tokenizer(start, return_tensors='pt', truncation=True).to(cfg['device'])
    inputs = inputs.to(cfg['device'])
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=2048,
            temperature=0.65,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.batch_decode(output)[0]
    return output

# Parse our text format back into a music21 score
def interpret(text):
    # measures = re.findall(r"\|.*?\|", text)

    outstream = m21.stream.Stream()
    note_match = re.compile(r"n\((?P<duration>[^,]*),(?P<notes>.*?)\)")
    # for measure in measures:
    matches = re.findall(note_match, text)
    for m in matches:
        try:
            duration = float(m[0])
            if duration == 0.:
                print("Model generated 0 duration, setting to 0.5 to continue...")
                duration = 0.5

            notes = [x.strip() for x in m[1].strip().split(' ')]

            m_duration = m21.duration.Duration(float(duration))
            m_notes = [m21.note.Note(x) for x in notes]

            m_chord = m21.chord.Chord(m_notes)
            m_chord.duration = m_duration
            outstream.append(m_chord)

        # Some of the mistakes are recoverable, but for now just drop failed notes.
        except Exception as e:
            print("Failed to build note, skipping. {}".format(e))

    return outstream

if __name__ == '__main__':
    with open('train_cfg.json', 'r') as f:
        cfg = json.load(f)
        print("Using config", cfg)

    output = infer(cfg)

    with open('out.txt', 'w') as f:
        f.write(output)

    # interpret(teststr).show()
    interpret(output).show()
