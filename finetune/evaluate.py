"""Сравнение трёх вариантов на вопросах, которых НЕ было в обучении (eval_questions.jsonl).

  base      — исходная модель без контекста (знает ли она резюме? нет, и не должна)
  base+rag  — исходная модель, которой в промпт кладут нужный фрагмент (как в чате resume-chat)
  tuned     — дообученная модель без контекста (знания «в весах»)

Метрики: «факт» — в ответе есть обязательная подстрока; «отказ» — на постороннее модель вежливо отказывает.
"""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import is_refusal, messages

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--lora", default=str(HERE / "out/lora"))
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--max-new", type=int, default=160)
ap.add_argument("--report", default=str(HERE / "out/eval.json"))
args = ap.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to(device).eval()
chunks = {c["id"]: c["text"] for c in json.load(open(HERE / "data/knowledge.json", encoding="utf-8"))}
qs = [json.loads(l) for l in open(HERE / "data/eval_questions.jsonl", encoding="utf-8")]


@torch.no_grad()
def answer(model, q, context=None):
    prompt = tok.apply_chat_template(messages(q, context=context), tokenize=False, add_generation_prompt=True)
    x = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**x, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run(model, name, rag=False):
    facts = ok_facts = refusals = ok_ref = 0
    rows = []
    for item in qs:
        ctx = "\n\n".join(chunks[i] for i in item.get("expect", [])) if rag and not item.get("refuse") else None
        a = answer(model, item["q"], ctx)
        if item.get("refuse"):
            refusals += 1
            hit = is_refusal(a)
            ok_ref += hit
        else:
            facts += 1
            hit = all(m.lower() in a.lower() for m in item["must"])
            ok_facts += hit
        rows.append({"q": item["q"], "answer": a, "ok": bool(hit), "refuse": bool(item.get("refuse"))})
    res = {"name": name, "facts": f"{ok_facts}/{facts}", "refusals": f"{ok_ref}/{refusals}", "rows": rows}
    print(f"{name:9s} факты {res['facts']:>6} | отказы {res['refusals']:>4}", flush=True)
    return res


results = [run(base, "base"), run(base, "base+rag", rag=True)]
if Path(args.lora).exists():
    tuned = PeftModel.from_pretrained(base, args.lora).eval()
    results.append(run(tuned, "tuned"))
Path(args.report).parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(args.report, "w"), ensure_ascii=False, indent=1)
