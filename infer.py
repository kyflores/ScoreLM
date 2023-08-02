import json
import sys
import re
import argparse

import music21 as m21
import transformers as tfs
import torch

def infer(opt, cfg):
    print("Loading model for inference.")
    tokenizer = tfs.AutoTokenizer.from_pretrained(
        cfg['model_name'],
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Need to manually enable use_cache. It must be disabled for training if
    # gradient_checkpointing is enabled, but if it's disabled, it slows generation
    config = tfs.AutoConfig.from_pretrained("score-lm")
    config.use_cache=True

    model = tfs.AutoModelForCausalLM.from_pretrained(
        "score-lm",
        config=config
    ).to(cfg['device'])
    model.eval()

    inputs = tokenizer(opt.prompt, return_tensors='pt', truncation=True).to(cfg['device'])
    inputs = inputs.to(cfg['device'])
    prompt_len_tok = inputs['input_ids'].shape[-1]

    model_text_size = config.hidden_size

    print("Prompt has size {}, leaving {} tokens for generation".format(
        prompt_len_tok,
        model_text_size - prompt_len_tok
    ))

    generation_cfg = tfs.GenerationConfig(
        do_sample=True,
        eos_token_id=model.config.eos_token_id,
        bos_token_id=model.config.bos_token_id,
        pad_token_id=model.config.eos_token_id,
        use_cache=True,
        max_new_tokens=(model_text_size - prompt_len_tok),
        temperature=opt.temperature,
        top_k=opt.top_k,
        top_p=opt.top_p,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_return_sequences=opt.generations
    )


    print("Loaded model. Will produce {} generations".format(opt.generations))
    output = []
    with torch.no_grad():
        # See https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate
        text = model.generate(
            **inputs,
            generation_config=generation_cfg,
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
            m_notes = []
            for n in notes:
                try:
                    tie = False
                    if n.startswith('T.'):
                        tie = True
                        n = n[2:]
                    note = m21.note.Note(n)
                    if tie:
                        note.tie = m21.tie.Tie('start')
                    m_notes.append(note)
                except Exception as e:
                    print("Generated invalid note \"{}\", exception {}.".format(n, e))

            m_duration = m21.duration.Duration(duration)

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
        "-t", "--temperature",
        type=float,
        help="Temperature to use for generation.",
        default=0.5
    )
    parser.add_argument(
        "--top_k",
        type=int,
        help="Value to use for topk sampling during generation.",
        default=50,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        help="Value to use for topp sampling during generation.",
        default=1.0,
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to a configuration file. Pass the same thing that was given to training.",
        default="configs/scorelm_1b_24GB_ds.json"
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
        output = infer(opt, cfg)
        for ix, v in enumerate(output):
            fname = "{}_{}.sclm".format(opt.scorefile, ix)
            with open(fname, 'w') as f:
                f.write(v)
    elif opt.mode == "post":
        with open(opt.scorefile, 'r') as f:
            text = f.read()
            interpret(text).show()
    elif opt.mode == "both":
        output = infer(opt, cfg)
        interpret(output[0]).show()
