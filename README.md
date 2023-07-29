# ScoreLM
Analyze musical scores with language models!

## Setup
First, create a new virtualenv or conda environment with python & pip.
Then:
```
pip install transformers datasets accelerate music21
```

Finally, note that the scripts generate some very large files, like datasets or model
weights. **Please DO NOT commit these to git!** The default names for these files are in the gitignore,
but please be careful.

## Data generation
Run `generate_data.py` to process the Bach Chorales into a text format.

## Training
Use `train.py` to train the model. This script finetunes Eleuther's Pythia model for the score text
format. Right now we use Pythia for a few reasons:
* Easy to access from Hugging Face, no downloading delta weights and merging like the official llama.
* Offers many scaling options for different hardware configurations. 70m is very manageable and can be trained
  on average hardware without `deepspeed`
* Pythia is trained on [The Pile](https://arxiv.org/pdf/2101.00027.pdf), which contains code text. Pretraining
  on code (or other formal languages like mathematics) is useful if the score text format resembles code.

Check out one of the [pythia model cards](https://huggingface.co/EleutherAI/pythia-12b-v0) for more info.

Parameters are selected in `train_cfg.json`.
In particular, check out these parameters:
```
model_name:
    Set by default to EleutherAI/pythia-70m-deduped, the smallest Pythia variant.
    If you have enough VRAM, you can select 160m, 410m, etc.
max_length:
    The sequence length the model can process, in tokens. Reduce if you're running out of VRAM.
batchsize:
    How many items to process in a batch. Use the largest batchsize your VRAM allows. Reduce if
    out of VRAM. When reducing batchsize, you may also need to reduce lr.
```
TODO:
* Support checkpointing and checkpoint loading.

### Deepspeed
Deepspeed implements optimizations for training that can significantly reduce the VRAM
requirement, at the cost of additional main RAM use. To use deepspeed,
```
accelerate launch --config_file accel_config.yaml train.py
```

## Inference
Use `infer.py` to output text with a trained model. This scripts loads the model
in a directory named `score-lm`, which is created by the training script when it finishes.
```
python infer.py "|"
```
The first argument is the text to start generating from. Since a `|` is the start of a new measure,
it is the usual way to start generation of a new score. You could also manually enter your own score,
and let it continue from that.

## Reconstructing the score.
TODO! This is about parsing the model's output text score back into music21 objects.
