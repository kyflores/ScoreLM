import json
import sys
import re
import argparse

import music21 as m21
import transformers as tfs
import torch

teststr = "|n(0.5, F#3 A3 C#4 F#4), n(0.5, F#3 B3 C#4 G#4)|_|n(0.25, F#3 C#4 F#4 A4), n(0.25, G#3 C#4 F#4 A4), n(0.25, A3 C#4 F#4 A4), n(0.25, B3 C#4 F#4 A4), n(0.5, C#4 E#4 G#4), n(0.5, C#3 B3 E#4 G#4), n(1.0, D3 A3 F#4), n(0.5, D3 B3 F#4), n(0.5, D3 B3 F#4 G#4)|_|n(0.5, C#3 C#4 F#4 A4), n(0.5, B2 C#4 F#4 A4), n(0.5, C#3 C#4 E#4 G#4), n(0.5, C#3 B3 E#4 G#4), n(1.0, F#2 A3 C#4 F#4), n(1.0, F#3 A3 F#4 C#5)|_|n(0.5, G#3 B3 F#4 B4), n(0.5, G#3 B3 E#4 B4), n(0.5, A3 C#4 F#4 A4), n(0.5, B3 C#4 F#4 A4), n(1.0, C#4 E#4 G#4), n(1.0, C#3 C#4 E#4 G#4)|_|n(0.5, F#3 C#4 F#4 A4), n(0.5, E3 C#4 F#4 A4), n(0.5, D3 D4 F#4 A4), n(0.5, C#3 D4 F#4 A4), n(0.5, D3 D4 F#4 B4), n(0.25, B2 E4 G#4 B4), n(0.25, B2 F#4 A4 B4), n(0.5, E3 E4 G#4 B4), n(0.5, E2 E3 D4 G#4 B4)|_|n(0.5, A2 C#4 G#4 C#5), n(0.5, A3 C#4 F#4 C#5), n(0.5, G#3 B3 E#4 C#5), n(0.5, F#3 A3 F#4 C#5), n(0.5, F#3 D4 G#4 B4), n(0.5, E#3 C#4 G#4 B4), n(0.5, F#3 C#4 G#4 A4), n(0.5, D3 D4 F#4 A4)|_|n(0.5, B2 D4 F#4 G#4), n(0.5, G#2 B3 F#4 G#4), n(0.5, C#3 G#3 E#4 G#4), n(0.5, C#3 C#4 E#4 G#4), n(1.0, F#2 A3 C#4 F#4), n(0.5, F#3 A3 F#4 C#5), n(0.5, E3 A3 F#4 C#5)|_|n(0.5, D3 B3 F#4 B4), n(0.5, C#3 C#4 E#4 B4), n(0.5, B#2 D#4 F#4 A4), n(0.5, B#2 D#4 F#4 G#4), n(1.0, C#3 C#4 E#4 G#4), n(0.5, A2 C#4 F#4 C#5), n(0.5, A2 C#4 E4 C#5)|_|n(0.5, B2 F#3 D#4 B4), n(0.5, C#3 F#3 E4 B4), n(0.5, D#3 B3 F#4 A4), n(0.5, B2 B3 D#4 A4), n(1.0, E3 B3 E4 G#4), n(0.5, E#3 C#4 G#4), n(0.5, C#3 C#4 E#4 G#4)|_|n(0.5, F#3 C#4 F#4 A4), n(0.5, F#2 F#3 A3 F#4 A4), n(0.5, F#3 D4 A4), n(0.5, D3 D4 F#4 A4), n(0.5, G3 D4 B4), n(0.5, G2 G3 B3 D4 B4), n(0.5, G#3 E4 B4), n(0.5, E3 E4 G#4 B4)|_|n(0.5, A3 E4 A4 C#5), n(0.5, A2 A3 C#4 G4 C#5), n(0.5, A#3 F#4 C#5), n(0.5, F#3 E4 F#4 C#5), n(0.5, B3 D4 F#4 B4), n(0.5, B2 B3 C#4 E#4 B4), n(0.5, B#3 D#4 F#4 A4), n(0.5, G#3 D#4 F#4 G#4)|_|n(0.5, C#4 F#4 G#4), n(0.25, G#3 B3 C#4 E#4 G#4), n(0.25, G#3 B3 C#4 D#4 G#4), n(0.5, C#3 C#4 E#4 G#4), n(0.5, C#3 B3 E#4 G#4), n(1.0, F#2 F#3 A3 C#4 F#4)|"

def infer(n, cfg, prompt=None):
    print("Loading model for inference.")
    tokenizer = tfs.AutoTokenizer.from_pretrained(
        cfg['model_name'],
        model_max_length=cfg['blocksize'],
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = tfs.AutoModelForCausalLM.from_pretrained(
        "score-lm",
    ).to(cfg['device'])

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(cfg['device'])
    inputs = inputs.to(cfg['device'])

    print("Loaded model. Will produce {} generations".format(n))
    output = []
    with torch.no_grad():
        # See https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate
        text = model.generate(
            **inputs,
            do_sample=True,
            # Generate up to this many tokens.
            max_new_tokens=512*2,
            # Set smaller for more predictable, regular generations. Set higher for more randomness.
            temperature=0.75,
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.0,
            length_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=n
        )
        print(text.shape)

        output = tokenizer.batch_decode(text)

    print(output)
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
            # Eval evalutes a str as if it were a piece of python source code.
            # This can handle triplets, which M21 notates as 1/3, or 1/6, etc...
            duration = eval(m[0])
            if duration <= (1/24):
                print("Model generated invalid duration, setting to 1.0 to continue...")
                duration = 1.0

            notes = [x.strip() for x in m[1].strip().split(' ')]

            m_duration = m21.duration.Duration(duration)
            m_notes = [m21.note.Note(x) for x in notes]

            m_chord = m21.chord.Chord(m_notes)
            m_chord.duration = m_duration
            outstream.append(m_chord)

        # Some of the mistakes are recoverable, but for now just drop failed notes.
        except Exception as e:
            print("Failed to build note, skipping. {}".format(e))

    return outstream

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate text with a trained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to a configuration file. Pass the same thing that was given to training.",
        default="train_cfg.json"
    )
    parser.add_argument(
        "-m", "--mode",
        type=str,
        choices=["infer", "post", "both"],
        default="both",
        help="Set to infer to generate text, or post to parse generated text"
            "Choosing \"both\" performs inference first, then post processing"
    )
    parser.add_argument(
        "-g", "--generations",
        type=int,
        default=1,
        help="Number of generations to produce in infer mode."
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default="|",
        help="Initial context for text generation."
    )
    parser.add_argument(
        "-s", "--scorefile",
        type=str,
        default="out",
        help="Filename to write to (infer) or read from (post)"
            "The extension .txt is appended to this name"
    )
    opt = parser.parse_args()

    with open(opt.config, 'r') as f:
        cfg = json.load(f)
        print("Using config", cfg)

    if opt.mode == "infer":
        output = infer(opt.generations, cfg, opt.prompt)
        for ix, v in enumerate(output):
            fname = "{}_{}.txt".format(opt.scorefile, ix)
            with open(fname, 'w') as f:
                f.write(v)
    elif opt.mode == "post":
        with open(opt.scorefile + ".txt", 'r') as f:
            text = f.read()
            interpret(text).show()
    elif opt.mode == "both":
        output = infer(1, cfg, opt.prompt)
        interpret(output[0]).show()
