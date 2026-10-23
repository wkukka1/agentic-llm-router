"""Cache-compatible prompt compression on long prompts.

Prompt caching (Anthropic, OpenAI, and Ollama's KV reuse) only pays for a prefix once
when the next request starts with the SAME tokens. So compression has to keep the
long, shared part of the prompt byte-stable across calls to keep the cache.

Setup: QuALITY long-story multiple-choice QA. Several questions are asked about each
story in a row, so the story is the reusable prefix. Strategies:

  full          story verbatim in the system message, question after it
  prefix@r      story compressed ONCE with LLMLingua-2 (deterministic) and reused
                byte-for-byte for every question -> cache-friendly
  query@r       per question, keep the story sentences most similar to the question
                (CPC-style, MiniLM encoder) -> question-aware, but the prefix changes
                every call, so nothing past the instruction can be cached

Cached tokens = longest shared token prefix with the previous request (Qwen2.5 chat
template), with a provider-style 1024-token minimum. Cost uses Anthropic-style pricing
(cache read 0.1x, cache write 1.25x) in units of one uncached input token. Answers come
from a local Ollama model, and its measured prefill time is the real-world cache check.

    python experiments/prompt_compression/cache_experiment.py --articles 20 --questions 10
"""
import argparse
import ast
import json
import re
import time
from collections import defaultdict

import numpy as np

OUT = "experiments/prompt_compression/results_cache_quality.json"
TOKENIZER = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM = ("Read the story, then answer the multiple-choice question with the letter of the "
          "correct option only.\n\n<story>\n{prefix}\n</story>")
LETTERS = "ABCD"


def shared_prefix_len(a, b) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def cache_accounting(token_lists, min_cache: int = 1024) -> list[dict]:
    """Per request: tokens read from cache (shared prefix with the previous request, if at
    least ``min_cache``), uncached tokens, and tokens written to cache (the part the next
    request reads that this one did not already read)."""
    cached = [0] * len(token_lists)
    for i in range(1, len(token_lists)):
        lcp = shared_prefix_len(token_lists[i - 1], token_lists[i])
        cached[i] = lcp if lcp >= min_cache else 0
    out = []
    for i, toks in enumerate(token_lists):
        nxt = cached[i + 1] if i + 1 < len(token_lists) else 0
        out.append({"cached": cached[i], "uncached": len(toks) - cached[i],
                    "written": max(0, nxt - cached[i])})
    return out


def input_cost(acc, read_mult: float = 0.1, write_mult: float = 1.25) -> float:
    """Input cost in uncached-token units: base price on uncached tokens, ``read_mult`` on
    cache reads, and the ``write_mult - 1`` premium on cache writes."""
    return float(sum(r["uncached"] + read_mult * r["cached"] + (write_mult - 1) * r["written"]
                     for r in acc))


def build_messages(prefix: str, question: str, options) -> list[dict]:
    opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))
    return [{"role": "system", "content": SYSTEM.format(prefix=prefix)},
            {"role": "user", "content": f"Question: {question}\n{opts}\nAnswer:"}]


def parse_choice(text: str) -> int:
    m = re.search(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])", text)
    return LETTERS.index(m.group(1).upper()) if m else -1


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def select_sentences(article: str, question: str, rate: float, embed) -> str:
    """Keep the sentences most cosine-similar to ``question`` within ``rate`` of the
    article's words, in their original order. ``embed(texts) -> [n, d]``."""
    sents = _sentences(article)
    vecs = np.asarray(embed(sents + [question]), np.float64)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    sims = vecs[:-1] @ vecs[-1]
    words = np.array([len(s.split()) for s in sents])
    budget = rate * words.sum()
    keep, used = [], 0
    for i in np.argsort(-sims, kind="stable"):
        if keep and used + words[i] > budget:
            continue
        keep.append(i)
        used += words[i]
    return " ".join(sents[i] for i in sorted(keep))


def compress_prefix(text: str, rate: float) -> str:
    """Question-agnostic LLMLingua-2 compression of the shared prefix (deterministic)."""
    from experiment import _compressor

    return _compressor().compress_prompt(text, rate=rate)["compressed_prompt"]


def _load_quality(n_articles: int, n_questions: int) -> list[dict]:
    from datasets import load_dataset

    by_article = defaultdict(list)
    for r in load_dataset("emozilla/quality", split="validation"):
        by_article[r["article"]].append(r)
    picked = [rows[:n_questions] for rows in by_article.values() if len(rows) >= n_questions][:n_articles]
    return [{"article_id": a, "article": rows[0]["article"], "question": r["question"],
             "options": (ast.literal_eval(r["options"]) if isinstance(r["options"], str)
                         else list(r["options"])),
             "answer": int(r["answer"]), "hard": str(r["hard"]) == "True"}
            for a, rows in enumerate(picked) for r in rows]


def _prefixes(items, rates, query_rate) -> tuple[dict, dict]:
    """``{strategy: [prefix per item]}`` plus compression wall time per strategy."""
    import torch
    from router.config import load_config
    from router.embeddings import load_encoder

    articles = {it["article_id"]: it["article"] for it in items}
    out, secs = {"full": [it["article"] for it in items]}, {"full": 0.0}
    for r in rates:
        t0 = time.time()
        comp = {a: compress_prefix(text, r) for a, text in articles.items()}
        out[f"prefix@{r:g}"] = [comp[it["article_id"]] for it in items]
        secs[f"prefix@{r:g}"] = time.time() - t0
    enc = load_encoder(load_config(), "retrieval")
    t0 = time.time()
    out[f"query@{query_rate:g}"] = [select_sentences(it["article"], it["question"], query_rate, enc.encode)
                                    for it in items]
    secs[f"query@{query_rate:g}"] = time.time() - t0
    import experiment
    experiment._compressor.cache_clear()
    del enc
    torch.cuda.empty_cache()   # leave the GPU to Ollama
    return out, secs


def main() -> None:
    import ollama
    from transformers import AutoTokenizer
    from training.cli import resolve, write_json

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--articles", type=int, default=20)
    ap.add_argument("--questions", type=int, default=10)
    ap.add_argument("--rates", default="0.5,0.33", help="LLMLingua-2 prefix rates")
    ap.add_argument("--query-rate", type=float, default=0.5)
    ap.add_argument("--min-cache", type=int, default=1024)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    items = _load_quality(args.articles, args.questions)
    print(f"{len(items)} questions over {len(set(it['article_id'] for it in items))} stories")
    prefixes, comp_secs = _prefixes(items, [float(r) for r in args.rates.split(",")], args.query_rate)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    def to_ids(m):
        text = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        return tok(text, add_special_tokens=False)["input_ids"]
    msgs = {s: [build_messages(p, it["question"], it["options"]) for p, it in zip(ps, items)]
            for s, ps in prefixes.items()}
    ids = {s: [to_ids(m) for m in ms] for s, ms in msgs.items()}
    longest = max(len(t) for ts in ids.values() for t in ts)
    num_ctx = int(np.ceil((longest + 64) / 1024) * 1024)
    print(f"longest prompt {longest} tokens -> num_ctx {num_ctx}")

    out = resolve(args.out)
    log_path = out.with_suffix(".rows.jsonl")   # one row per answered question; a rerun resumes from it
    done = {}
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[(row["strategy"], row["i"])] = row
        print(f"resuming: {len(done)} answers already in {log_path.name}")

    client = ollama.Client()
    rows, summary = [], {}
    for s, ms in msgs.items():
        print(f"\n--- {s} ---")
        acc = cache_accounting(ids[s], args.min_cache)
        for i, (m, it, a) in enumerate(zip(ms, items, acc)):
            if (s, i) in done:
                rows.append(done[(s, i)])
                continue
            r = client.chat(model=args.model, messages=m,
                            options={"num_ctx": num_ctx, "temperature": 0, "num_predict": 4})
            pick = parse_choice(r.message.content)
            row = {"strategy": s, "i": i, "article_id": it["article_id"], "question": it["question"],
                   "hard": it["hard"], "answer": it["answer"], "pick": pick,
                   "correct": pick == it["answer"], "tokens": len(ids[s][i]),
                   "ollama_prompt_tokens": r.prompt_eval_count,
                   "prefill_ms": r.prompt_eval_duration / 1e6, **a}
            rows.append(row)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(ms)}")
        sr = [x for x in rows if x["strategy"] == s]
        correct = np.array([x["correct"] for x in sr], float)
        summary[s] = {
            "accuracy": float(correct.mean()), "se": float(correct.std(ddof=1) / np.sqrt(len(correct))),
            "mean_tokens": float(np.mean([x["tokens"] for x in sr])),
            "cached_share": float(sum(x["cached"] for x in sr) / sum(x["tokens"] for x in sr)),
            "cost_no_cache": float(sum(x["tokens"] for x in sr)),
            "cost_with_cache": input_cost(sr), "mean_prefill_ms": float(np.mean([x["prefill_ms"] for x in sr])),
            "compression_s": comp_secs[s],
            "token_count_check": float(np.mean([x["ollama_prompt_tokens"] / x["tokens"] for x in sr])),
        }

    full = np.array([x["correct"] for x in rows if x["strategy"] == "full"], float)
    base = summary["full"]["cost_no_cache"]
    print(f"\n{len(full)} questions, model {args.model}. Cost = input tokens priced Anthropic-style "
          f"(read 0.1x, write 1.25x), relative to `full` without caching.\n")
    print("| strategy | accuracy | vs full (paired) | avg prompt tokens | cached share | cost, no cache "
          "| cost, with cache | mean prefill ms |\n|---|---|---|---|---|---|---|---|")
    for s, m in summary.items():
        this = np.array([x["correct"] for x in rows if x["strategy"] == s], float)
        d = this - full
        m["paired_diff_vs_full"] = float(d.mean())
        m["paired_diff_se"] = float(d.std(ddof=1) / np.sqrt(len(d)))
        print(f"| {s} | {m['accuracy']:.3f} +/- {m['se']:.3f} | {d.mean():+.3f} +/- {m['paired_diff_se']:.3f} "
              f"| {m['mean_tokens']:.0f} | {m['cached_share']:.0%} | {m['cost_no_cache'] / base:.2f} "
              f"| {m['cost_with_cache'] / base:.2f} | {m['mean_prefill_ms']:.0f} |")
    print("\ntoken count check (ollama / HF template): "
          + ", ".join(f"{s} {m['token_count_check']:.3f}" for s, m in summary.items()))

    import pandas as pd
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
    write_json(out, {"model": args.model, "n_questions": len(full), "articles": args.articles,
                     "questions_per_article": args.questions, "min_cache": args.min_cache,
                     "num_ctx": num_ctx, "summary": summary})


if __name__ == "__main__":
    main()
