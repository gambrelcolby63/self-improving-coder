"""LoRA fine-tune of Qwen2.5-Coder-1.5B-Instruct on the experience library export.

NOT run on the dev box (CPU only). Run on a free Kaggle (T4/P100) or Colab (T4) GPU:
    pip install -q "transformers>=4.45" "peft>=0.13" "trl>=0.20" datasets accelerate
    python scripts/finetune_lora.py --data data/finetune/experience_sft.jsonl --out lora-out

The JSONL contains only TRAIN-split problems (HumanEval train split + self-written
train tasks) whose solutions passed their tests, plus repair traces (bad code ->
sandbox error -> fixed code).  Eval-split tasks are never included, so re-running
`evals/run_eval.py` with the fine-tuned model is a fair comparison.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/finetune/experience_sft.jsonl")
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--out", default="lora-out")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--merge", action="store_true", help="also save merged fp16 weights for GGUF export")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    eval_ids = set(json.load(open("evals/splits.json"))["eval"]) if __import__("os").path.exists("evals/splits.json") else set()
    rows = [r for r in rows if r["meta"]["id"] not in eval_ids]  # belt and braces
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    print(f"{len(ds)} training conversations")

    tok = AutoTokenizer.from_pretrained(a.base)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16 if bf16 else torch.float16,
                                                 device_map="auto")
    peft_cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
                    per_device_train_batch_size=2, gradient_accumulation_steps=4, lr_scheduler_type="cosine",
                    warmup_ratio=0.05, logging_steps=5, save_strategy="no", bf16=bf16, fp16=not bf16,
                    max_length=2048, assistant_only_loss=False, report_to=[])
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
    tr.train()
    tr.save_model(a.out)
    print("saved LoRA adapter to", a.out)
    if a.merge:
        merged = tr.model.merge_and_unload()
        merged.save_pretrained(a.out + "-merged")
        tok.save_pretrained(a.out + "-merged")
        print("saved merged model to", a.out + "-merged")


if __name__ == "__main__":
    main()
