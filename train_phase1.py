"""Train LM 1 on data/train.bin (random 512-token windows), FP16 mixed precision for the RTX 2060 (Turing: no BF16).

AdamW (0.9, 0.95), weight decay 0.1 on matrices only, warmup then cosine to 10% of peak LR, grad clip 1.0.
The run sizes itself to --hours (measured on warm steps); evaluates val loss + writes sample generations every
--eval_every steps; keeps best.pt (by val loss) and latest.pt (full state for --resume). Logged to MLflow.
    py -3.12 train.py --hours 4
    py -3.12 train.py --steps 60 --smoke
    py -3.12 train.py --resume
"""
import os, time, math, json, argparse, numpy as np, torch
from tokenizers import Tokenizer
from model import GPT, Config, count_params

H = os.path.dirname(os.path.abspath(__file__)); CK = os.path.join(H, "checkpoints"); os.makedirs(CK, exist_ok=True)
DEV = torch.device("cuda")
ap = argparse.ArgumentParser()
ap.add_argument("--hours", type=float, default=None); ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--micro", type=int, default=32); ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--eval_every", type=int, default=500); ap.add_argument("--resume", action="store_true")
ap.add_argument("--smoke", action="store_true"); ap.add_argument("--data", default=os.path.join(H, "data"))
args = ap.parse_args()
PROMPTS = ["Once upon a time", "The human heart is", "In mathematics, a prime number", "The city of Chicago"]


def batches(split, cfg):
    data = np.memmap(os.path.join(args.data, f"{split}.bin"), dtype=np.uint16, mode="r")
    while True:
        ix = np.random.randint(0, len(data) - cfg.block - 1, args.micro)
        x = torch.from_numpy(np.stack([data[i:i + cfg.block] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i + 1:i + 1 + cfg.block] for i in ix]).astype(np.int64))
        yield x.pin_memory().to(DEV, non_blocking=True), y.pin_memory().to(DEV, non_blocking=True)


@torch.no_grad()
def evaluate(model, it, n=40):
    model.eval(); tot = 0.0
    for _ in range(n):
        x, y = next(it)
        with torch.autocast("cuda", dtype=torch.float16): _, loss = model(x, y)
        tot += loss.item()
    model.train(); return tot / n


def samples(model, tok, eot):
    model.eval(); out = []
    for p in PROMPTS:
        idx = torch.tensor([tok.encode(p).ids], device=DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            g = model.generate(idx, 80, eot=eot)
        out.append(tok.decode(g[0].tolist()))
    model.train(); return out


def keep_awake():
    if os.name == "nt":
        import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def main():
    import mlflow
    keep_awake(); torch.manual_seed(0); np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    meta = json.load(open(os.path.join(args.data, "meta.json")))
    tok = Tokenizer.from_file(os.path.join(H, "tokenizer.json")); eot = meta["eot"]
    cfg = Config(vocab=meta["vocab"]); model = GPT(cfg).to(DEV)
    print(f"device: {torch.cuda.get_device_name(0)} | params {count_params(model) / 1e6:.1f}M | "
          f"{args.micro * args.accum * cfg.block:,} tokens/step", flush=True)
    decay = [p for p in model.parameters() if p.dim() >= 2]; nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    scaler = torch.amp.GradScaler("cuda")
    step, best, total, run_id = 0, float("inf"), args.steps, None
    if args.resume and os.path.exists(os.path.join(CK, "latest.pt")):
        ck = torch.load(os.path.join(CK, "latest.pt"), map_location=DEV, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        step, best, total, run_id = ck["step"], ck["best"], ck["total"], ck.get("run_id")
        print(f"resumed at step {step}/{total}", flush=True)
    tr, va = batches("train", cfg), batches("val", cfg)
    mlflow.set_tracking_uri("sqlite:///" + os.path.join(H, "mlflow.db").replace("\\", "/"))
    mlflow.set_experiment("lm1-tiny-gpt")
    t0 = time.time(); start = step; tpm = None
    with mlflow.start_run(run_id=run_id, run_name="smoke" if args.smoke else None) as run:
        if run_id is None:
            mlflow.log_params({**cfg.__dict__, "params_M": round(count_params(model) / 1e6, 2), "lr": args.lr,
                               "tokens_per_step": args.micro * args.accum * cfg.block, "hours": args.hours,
                               "train_tokens": meta["train_tokens"], "precision": "fp16+GradScaler"})
        while total is None or step < total:
            if step < args.warmup: lr = args.lr * (step + 1) / args.warmup
            elif total is not None:
                pr = min((step - args.warmup) / max(total - args.warmup, 1), 1.0)
                lr = args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * pr)))
            else: lr = args.lr
            for g in opt.param_groups: g["lr"] = lr
            for _ in range(args.accum):
                x, y = next(tr)
                with torch.autocast("cuda", dtype=torch.float16): _, loss = model(x, y)
                scaler.scale(loss / args.accum).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
            if total is None and step == start + 50: torch.cuda.synchronize(); tpm = time.time()
            if total is None and step == start + 250:
                torch.cuda.synchronize(); per = (time.time() - tpm) / 200
                total = int((args.hours * 3600 - (time.time() - t0)) * 0.93 / per) + step
                print(f"{per:.3f} s/step -> {total} steps ({total * args.micro * args.accum * cfg.block / 1e6:.0f}M "
                      f"tokens) for {args.hours} h", flush=True); mlflow.log_param("total_steps", total)
            if step % 50 == 0: print(f"step {step}/{total}  loss {loss.item():.3f}  lr {lr:.2e}", flush=True)
            if step % args.eval_every == 0 or step == total:
                vl = evaluate(model, va, 10 if args.smoke else 40)
                mlflow.log_metrics({"train_loss": loss.item(), "val_loss": vl, "val_ppl": math.exp(vl), "lr": lr}, step=step)
                s = samples(model, tok, eot)
                with open(os.path.join(H, "samples.txt"), "a", encoding="utf-8") as f:
                    f.write(f"\n===== step {step} | val loss {vl:.3f} | ppl {math.exp(vl):.1f} =====\n" + "\n---\n".join(s) + "\n")
                print(f"EVAL step {step}: val loss {vl:.3f} | ppl {math.exp(vl):.1f} | {(time.time() - t0) / 3600:.2f} h", flush=True)
                if vl < best:
                    best = vl; torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step, "val_loss": vl},
                                          os.path.join(CK, "best.pt")); print("  saved best", flush=True)
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                            "step": step, "best": best, "total": total, "run_id": run.info.run_id}, os.path.join(CK, "latest.pt"))
        mlflow.log_metric("best_val_loss", best)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
