import transformers as tfs
import datasets as ds

import json
import torch

# Based on https://huggingface.co/docs/transformers/tasks/language_modeling

class JsonlGenerator:
    def __init__(self, cfg, tok, fname):
        # Lazy loader pulls the entire thing into RAM
        # Won't work if the dataset gets too large.
        with open(fname, 'r') as f:
            lines = f.readlines()
            self.text = lines

        self.tokenizer = tok

    def _generate(self):
        for v in self.text:
            x = json.loads(v)
            yield x

    def preprocess(self, x):
        # TODO: we need to split up long lines, not truncate.
        # There's an example of something similar in the huggingface tutorial
        text = self.tokenizer(x['text'], truncation=True)

        return text

    def get_dsets(self):
        chorale_dset = ds.Dataset.from_generator(self._generate)
        self.dset_split = chorale_dset.train_test_split(test_size=0.2)

        self.tokenized_train = self.dset_split['train'].shuffle().map(
            self.preprocess,
            batched=True,
        )
        self.tokenized_val = self.dset_split['test'].shuffle().map(
            self.preprocess,
            batched=True,
        )

        return self.tokenized_train, self.tokenized_val

class ScorePredictorModel:
    def __init__(self, cfg, tok):
        self.cfg = cfg
        self.tokenizer = tok

        self.model = tfs.AutoModelForCausalLM.from_pretrained(
            cfg['model_name'],
        ).to(self.cfg['device'])
        self.collator = tfs.DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

    def train(self, dset_train, dset_val):
        args = tfs.TrainingArguments(
            output_dir='./finetune',
            optim='adamw_torch',
            learning_rate=float(self.cfg['lr']),
            num_train_epochs=self.cfg['epochs'],
            per_device_train_batch_size=self.cfg['batchsize'],
            per_device_eval_batch_size=self.cfg['batchsize'],
            weight_decay=0.01,
            save_strategy='steps',
            save_steps=self.cfg['save_steps'],
            evaluation_strategy='epoch',
            push_to_hub=False,
            report_to='none',
            # bf16=True,
            # no_cuda=True,
            # use_ipex=True
        )
        # Thank you random person https://discuss.huggingface.co/t/training-loss-0-0-validation-loss-nan/27950
        self.trainer = tfs.Trainer(
            model=self.model,
            args=args,
            train_dataset=dset_train,
            eval_dataset=dset_val,
            tokenizer=self.tokenizer,
            data_collator=self.collator
        )
        self.trainer.train()

    def evaluate(self):
        import math
        eval_results = self.trainer.evaluate()
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True)
        inputs = inputs.to(self.cfg['device'])
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=1024,
                temperature=0.25,
                top_k=60,
                pad_token_id=tokenizer.eos_token_id,
            )
        output = self.tokenizer.batch_decode(output)[0]
        print(output)

    def save(self, name):
        self.trainer.save_model(name)

if __name__ == '__main__':
    with open('train_cfg.json', 'r') as f:
        cfg = json.load(f)
        print("Using config", cfg)

    tokenizer = tfs.AutoTokenizer.from_pretrained(
        cfg['model_name'],
        model_max_length=cfg['max_length'],
    )
    tokenizer.pad_token = tokenizer.eos_token

    t, v = JsonlGenerator(cfg, tokenizer, 'data.txt').get_dsets()
    mdl = ScorePredictorModel(cfg, tokenizer)
    mdl.train(t, v)
    mdl.save('score-lm')
    mdl.evaluate()
    mdl.predict('|')
