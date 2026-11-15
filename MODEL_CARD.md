---
license: mit
language:
- en
library_name: pytorch
pipeline_tag: text-generation
tags:
- gpt
- from-scratch
- small-language-model
- educational
datasets:
- HuggingFaceFW/fineweb-edu
- HuggingFaceTB/smollm-corpus
- roneneldan/TinyStories
- wikimedia/wikipedia
---

# Tiny GPT (LM 1)

> **AI disclosure:** This project was built with AI assistance (Claude, by Anthropic) for writing code and
> documentation. All creative and technical decisions (model design, training data, evaluation) are mine, and I
> reviewed everything before publishing.

💻 [Code on GitHub](https://github.com/Stavion-Colquitt/Tiny-GPT) (tokenizer, model, training loop)

My first language model: a 20.7M-parameter GPT written from scratch in PyTorch and trained from zero on a single
**RTX 2060 (6 GB)** in about 9.5 hours, on 505M tokens of openly licensed English text. No pretrained weights, no
`transformers` library: the tokenizer, model, training loop and sampler are all in this repo.

---

## Results

Validation loss on 2.5M held-out tokens (every 200th document of the mix, never trained on):

| Run | Steps | Tokens seen | Time on the 2060 | Best val loss | Perplexity |
|-----|-------|-------------|------------------|---------------|------------|
| Phase 1 (cosine, peak LR 1e-3) | 9,644 | 316M | 3.75 h | 3.057 | 21.3 |
| Extension (re-warm to 3e-4, cosine to 3e-5, new seed) | 21,877 | 717M total | +5.77 h | **2.922** | **18.6** |

The extension bought about 0.13 nats. Most of it came early: from step 19,000 on, val loss sat between 2.92 and
2.97, with the best value at the very end of the cosine (step 21,877). At this size and data budget the next real
gains come from more data or a bigger model, not more steps.

Sample from the final checkpoint (step 21,877, temperature 0.8, top-k 50), prompt in bold:

> **Once upon a time**, there was a big, ugly cloud. The cloud lived in a big place called a volcano. The volcano
> was not pretty. The volcano was very old and quiet. The people in the place would always say "hello" to the
> volcano. They liked the volcano because it made them feel nice.

It writes fluent short stories and Wikipedia-style sentences; facts and math are made up (see *Limitations*).
`samples.txt` has generations from every evaluation during training.

## Model

| | |
|---|---|
| Architecture | decoder-only transformer, pre-norm, 8 layers, 6 heads, d_model 384, GELU MLP (4x) |
| Context | 512 tokens, learned positional embeddings |
| Tokenizer | byte-level BPE, 16,384 tokens, trained on the corpus (`tokenizer.json`) |
| Parameters | 20.7M (token embedding tied to the output head) |
| Attention | PyTorch `scaled_dot_product_attention` (causal) |

## Data (505M training tokens)

| Source | Tokens | License |
|--------|--------|---------|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) sample-10BT, shard 0 | 229.5M | ODC-By 1.0 |
| [Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) (SmolLM corpus), shard 0 | 125.4M | ODC-By 1.0 |
| [TinyStories V2](https://huggingface.co/datasets/roneneldan/TinyStories) (GPT-4 stories) | 75.4M | CDLA-Sharing 1.0 |
| [Simple English Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) 2023-11-01 | 75.2M | CC BY-SA 3.0 |

## Training

- AdamW (0.9, 0.95), weight decay 0.1 on matrices only, gradient clipping 1.0
- 32 x 2 micro-batches of 512 tokens = 32,768 tokens per step; FP16 autocast + GradScaler (Turing has no BF16)
- Phase 1: 500 warmup steps, cosine from 1e-3 to 1e-4, sized to a 4-hour budget from measured step time
- Extension: resume from the last state, 300-step re-warm to 3e-4, cosine to 3e-5 over 6 hours, new data seed
- Evaluated every 500 steps (val loss + samples), best checkpoint kept by val loss, logged to MLflow

## Use it

This is a plain PyTorch model (not a `transformers` model). Download the repo and run the included script:

```
pip install torch tokenizers safetensors huggingface_hub
huggingface-cli download Stavion-Colquitt/Tiny-GPT-LM1 --local-dir tiny-gpt-lm1
cd tiny-gpt-lm1
python generate.py "Once upon a time" --tokens 120
```

Files: `model.safetensors` (weights, best checkpoint: step 21,877), `config.json`, `tokenizer.json`, `model.py`,
`generate.py`. Runs fine on a CPU.

## Limitations

A 20M-parameter model trained on half a billion tokens: it is a learning project, not an assistant. It is not
instruction-tuned, makes up facts, loses the thread after a few sentences, and inherits whatever biases are in
the training text. Don't use it for anything that matters.

## License

Code: MIT. The training data keeps its own licenses (table above).
