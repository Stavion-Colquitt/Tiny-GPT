"""Build the LM 1 training corpus: train a 16k byte-level BPE tokenizer, then tokenize the mix to fixed budgets.

Mix (target ~500M tokens): FineWeb-Edu 225M | Cosmopedia v2 125M | TinyStories V2 75M | Simple Wikipedia (all, cap 75M)
Every 200th document goes to validation. Output: data/train.bin, data/val.bin (uint16), tokenizer.json, data/meta.json
"""
import os, json, time, numpy as np, pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

H = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(H, "data", "raw"); OUT = os.path.join(H, "data")
VOCAB, EOT = 16384, "<|endoftext|>"
SOURCES = [  # name, loader key, path, token budget
    ("fineweb_edu", "parquet", "sample/10BT/000_00000.parquet", 225_000_000),
    ("cosmopedia_v2", "parquet", "cosmopedia-v2/train-00000-of-00104.parquet", 125_000_000),
    ("tinystories_v2", "txt", "TinyStoriesV2-GPT4-train.txt", 75_000_000),
    ("simple_wikipedia", "parquet", "20231101.simple/train-00000-of-00001.parquet", 75_000_000),
]


def docs(kind, rel):
    path = os.path.join(RAW, rel)
    if kind == "parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=["text"]):
            for t in batch.column(0).to_pylist():
                if t and t.strip(): yield t.strip()
    else:  # TinyStories: stories separated by <|endoftext|>
        buf = []
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if EOT in line:
                    head = line.split(EOT)[0]
                    if head.strip(): buf.append(head)
                    s = "".join(buf).strip(); buf = []
                    if s: yield s
                else:
                    buf.append(line)


def train_tokenizer():
    def sample():  # ~40 MB of text per source, in proportion-ish, for the BPE merges
        for _, kind, rel, _ in SOURCES:
            n = 0
            for d in docs(kind, rel):
                yield d; n += len(d)
                if n > 40_000_000: break
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=[EOT], min_frequency=2,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(sample(), tr)
    tok.save(os.path.join(H, "tokenizer.json")); return tok


def main():
    t0 = time.time(); tok = train_tokenizer(); eot = tok.token_to_id(EOT)
    print(f"tokenizer: {tok.get_vocab_size()} tokens in {time.time() - t0:.0f} s", flush=True)
    train, val, meta = [], [], {"vocab": tok.get_vocab_size(), "eot": eot, "sources": {}}
    for name, kind, rel, budget in SOURCES:
        n_tr = n_va = n_docs = 0; batch = []

        def flush(b):
            nonlocal n_tr, n_va
            for i, enc in enumerate(tok.encode_batch(b)):
                ids = np.array(enc.ids + [eot], dtype=np.uint16)
                if (n_docs_base + i) % 200 == 0: val.append(ids); n_va += len(ids)
                else: train.append(ids); n_tr += len(ids)

        for d in docs(kind, rel):
            batch.append(d); n_docs += 1
            if len(batch) == 4096:
                n_docs_base = n_docs - len(batch); flush(batch); batch = []
                if n_tr >= budget: break
        if batch and n_tr < budget:
            n_docs_base = n_docs - len(batch); flush(batch)
        meta["sources"][name] = {"train_tokens": n_tr, "val_tokens": n_va, "docs": n_docs}
        print(f"{name}: {n_tr / 1e6:.1f}M train / {n_va / 1e6:.2f}M val tokens from {n_docs:,} docs "
              f"({time.time() - t0:.0f} s)", flush=True)
    for split, parts in (("train", train), ("val", val)):
        arr = np.concatenate(parts); arr.tofile(os.path.join(OUT, f"{split}.bin"))
        meta[f"{split}_tokens"] = int(len(arr))
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    print(f"PREP DONE: {meta['train_tokens'] / 1e6:.0f}M train, {meta['val_tokens'] / 1e6:.1f}M val tokens "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
