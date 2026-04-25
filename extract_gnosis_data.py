import os
import sys

# CRITICAL: Force Python to use the local Gnosis transformers branch.
# We must clear any already-loaded transformers from the Anaconda environment
# so that the local branch's Qwen3ForCausalLM (which has `_should_stop`) is used.
local_transformers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "transformers/src"))
if local_transformers_path not in sys.path:
    sys.path.insert(0, local_transformers_path)
for key in list(sys.modules.keys()):
    if key.startswith("transformers"):
        del sys.modules[key]

import argparse
import pandas as pd
import torch

# Bypass cuDNN initialization bugs in the current conda environment
torch.backends.cudnn.enabled = False

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
from tqdm import tqdm

from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
import transformers

transformers.Qwen3ForCausalLM = Qwen3ForCausalLM

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def has_correctness_head(model) -> bool:
    return (
        hasattr(model, "_should_stop")
        and hasattr(model, "stop_head")
    )

@torch.no_grad()
def extract_features(model, tokenizer, texts, device, max_len=1024):
    enc = tokenizer(texts, return_tensors="pt", truncation=True, max_length=max_len, padding=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_attentions=True,
        output_hidden_states=False,
    )
    hidden_states = out.last_hidden_state
    logits = model.lm_head(hidden_states)

    probs = torch.softmax(logits, dim=-1)
    token_probs = probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    attn_stack = out.attentions

    scores = model._should_stop(
        last_hidden=hidden_states,
        attn_stack=attn_stack,
        token_probs=token_probs,
        mask=attention_mask.float(),
        input_ids=input_ids,
    )
    return scores.squeeze(-1).float().cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset_id", type=str, default="openai/gsm8k")
    parser.add_argument("--dataset_config", type=str, default="main")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="data/extracted")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Loading model: {args.model_id}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        use_cache=False,
    ).to(device).eval()

    if rank == 0:
        if not has_correctness_head(model):
            print("WARNING: Model does not have `_should_stop` / correctness head.")
        
    ds = load_dataset(args.dataset_id, args.dataset_config, split=args.split)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, collate_fn=lambda x: x)

    results = []
    
    # Dummy formatting logic (adjust based on dataset)
    for batch in tqdm(dataloader, disable=(rank != 0)):
        texts = []
        for ex in batch:
            q = ex.get("question", "")
            a = ex.get("answer", "")
            prompt = tokenizer.apply_chat_template([
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ], tokenize=False)
            texts.append(prompt)
            
        try:
            scores = extract_features(model, tokenizer, texts, device)
            for ex, score in zip(batch, scores):
                results.append({
                    "question": ex.get("question", ""),
                    "answer": ex.get("answer", ""),
                    "gnosis_score": float(score)
                })
        except Exception as e:
            if rank == 0:
                print(f"Error during extraction: {e}")

    df = pd.DataFrame(results)
    output_file = os.path.join(args.output_dir, f"shard_{rank}.parquet")
    df.to_parquet(output_file)
    
    if rank == 0:
        print(f"Finished extraction. Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
