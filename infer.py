import json
import sys

import transformers as tfs
import torch

if __name__ == '__main__':
    with open('train_cfg.json', 'r') as f:
        cfg = json.load(f)
        print("Using config", cfg)

    tokenizer = tfs.AutoTokenizer.from_pretrained(
        cfg['model_name'],
        model_max_length=cfg['max_length'],
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = tfs.AutoModelForCausalLM.from_pretrained(
        "score-lm",
    ).to(cfg['device'])

    inputs = tokenizer(sys.argv[1], return_tensors='pt', truncation=True).to(cfg['device'])
    inputs = inputs.to(cfg['device'])
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=1024,
            temperature=0.7,
            top_k=60,
            pad_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.batch_decode(output)[0]
    print(output)
