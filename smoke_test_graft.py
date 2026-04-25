import torch
import time
import sys
import os

# Bypass broken cuDNN in this conda environment (same fix as extract_gnosis_data.py)
torch.backends.cudnn.enabled = False

# Force local transformers
local_transformers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "transformers/src"))
if local_transformers_path not in sys.path:
    sys.path.insert(0, local_transformers_path)
for key in list(sys.modules.keys()):
    if key.startswith("transformers"):
        del sys.modules[key]

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

print("=== Task 3: Local Smoke Test (Llama Graft) ===")

config = LlamaConfig(
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=2,  # use 2 layers to save VRAM
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=4096,
    _attn_implementation="eager",
    use_stop_attn=True,
    use_stop_conf=False,
    use_stop_hid=True,
    stop_att_dim=256,
    stop_hid_dim=256
)

print("[1] Instantiating Patched LlamaForCausalLM...")
model = LlamaForCausalLM(config).cuda().bfloat16()
model.train()

B = 2
S = 64
input_ids = torch.randint(0, config.vocab_size, (B, S), device="cuda")
attention_mask = torch.ones((B, S), device="cuda")
labels = torch.randint(0, config.vocab_size, (B, S), device="cuda")
correctness_label = torch.randint(0, 2, (B, 1), device="cuda").float()

print(f"[2] Running Forward Pass (B={B}, S={S})...")
torch.cuda.reset_peak_memory_stats()
start_mem = torch.cuda.memory_allocated()
start_t = time.time()

output = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    correctness_label=correctness_label
)

end_t = time.time()
peak_mem = torch.cuda.max_memory_allocated()

print("\n--- Results ---")
print(f"Logits shape: {output.logits.shape}")
print(f"Loss value: {output.loss.item()}")
if hasattr(output, "stop_prob"):
    print(f"Stop Prob shape: {output.stop_prob.shape}")
    print(f"Stop Prob value: {output.stop_prob}")
else:
    print("ERROR: stop_prob missing!")

print(f"Forward Latency: {(end_t - start_t)*1000:.2f} ms")
print(f"VRAM Peak: {peak_mem / 1024**2:.2f} MB")
print("Smoke test completed successfully!")
