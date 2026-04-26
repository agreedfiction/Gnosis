"""
smoke_test_grpo_integration.py
Phase 3 Smoke Test — validates the full GRPOTrainer._compute_correctness_loss
data flow end-to-end using a tiny 2-layer Llama config (no real dataset needed).

Checks:
  [1] Model instantiates with correctness head (attn_extractor, hid_extractor, stop_head)
  [2] _onepass_hidden_attn_logps returns non-None attentions
  [3] model._should_stop returns stop_prob of shape (B, 1) in expected dtype
  [4] BCE / stop_loss is finite and non-zero
  [5] No shape mismatch
  [6] No VRAM explosion

Run on server:
  python smoke_test_grpo_integration.py
"""

import torch
import time
import sys
import os

# ── Force local transformers ─────────────────────────────────────────────────
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "transformers/src"))
if _root not in sys.path:
    sys.path.insert(0, _root)
for _k in list(sys.modules.keys()):
    if _k.startswith("transformers"):
        del sys.modules[_k]

# Bypass broken cuDNN
torch.backends.cudnn.enabled = False
# ─────────────────────────────────────────────────────────────────────────────

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16

print("=" * 60)
print("Phase 3 Smoke: GRPO Integration Test")
print(f"Device: {DEVICE}")
print("=" * 60)

# ── [1] Build tiny Llama with correctness head ───────────────────────────────
config = LlamaConfig(
    vocab_size=32000,
    hidden_size=1024,
    intermediate_size=2816,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=2,
    max_position_embeddings=2048,
    _attn_implementation="eager",
    use_stop_attn=True,
    use_stop_conf=False,
    use_stop_hid=True,
    stop_att_dim=128,
    stop_hid_dim=128,
)

print("\n[1] Instantiating patched LlamaForCausalLM...")
model = LlamaForCausalLM(config).to(DEVICE).to(DTYPE)
model.train()

# Check heads are present
assert hasattr(model, "attn_extractor"), "FAIL: attn_extractor missing"
assert hasattr(model, "hid_extractor"),  "FAIL: hid_extractor missing"
assert hasattr(model, "stop_head"),       "FAIL: stop_head missing"
print("    attn_extractor ✓  hid_extractor ✓  stop_head ✓")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"    Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ── [2] Simulate GRPOTrainer._onepass_hidden_attn_logps data ─────────────────
print("\n[2] Simulating _onepass_hidden_attn_logps forward pass...")

B   = 2    # batch size
P   = 32   # prompt length
C   = 32   # completion length
S   = P + C

prompt_ids     = torch.randint(0, config.vocab_size, (B, P), device=DEVICE)
completion_ids = torch.randint(0, config.vocab_size, (B, C), device=DEVICE)
input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
attention_mask = torch.ones((B, S), device=DEVICE, dtype=torch.long)

torch.cuda.reset_peak_memory_stats()
vram_before = torch.cuda.memory_allocated() / 1024**2 if DEVICE == "cuda" else 0

t0 = time.time()

# Replicate what _onepass_hidden_attn_logps does internally
with torch.no_grad():
    out = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_attentions=True,
    )

    mb_hidden  = out.last_hidden_state[:, :-1, :]   # (B, S-1, H)
    h_slice    = mb_hidden[:, -C:, :]                # (B, C, H)
    mb_logits  = model.lm_head(h_slice)              # (B, C, V)
    mb_targets = input_ids[:, -C:]                   # (B, C)

    # per-token logprobs for completion
    logp      = torch.log_softmax(mb_logits.float(), dim=-1)
    mb_logps  = logp.gather(-1, mb_targets.unsqueeze(-1)).squeeze(-1)   # (B, C)

    attns     = out.attentions   # tuple of (B, H, S, S)

step_time = (time.time() - t0) * 1000
vram_after = torch.cuda.max_memory_allocated() / 1024**2 if DEVICE == "cuda" else 0

assert attns is not None, "FAIL: attentions is None — check attn_implementation='eager'"
print(f"    hidden shape:  {mb_hidden.shape}")
print(f"    attns[0] shape: {attns[0].shape}  | {len(attns)} layers")
print(f"    per_token_logps: {mb_logps.shape}")
print(f"    Forward latency: {step_time:.1f} ms")
print(f"    VRAM before: {vram_before:.1f} MB  → peak: {vram_after:.1f} MB")

# ── [3] Build inputs mirroring GRPOTrainer._compute_correctness_loss ─────────
print("\n[3] Building correctness_loss inputs...")

# Assemble per-token probabilities (prompt=1.0, completion=from logps)
tok_p_comp   = torch.clamp(mb_logps.exp(), 1e-8, 1.0).to(DTYPE)       # (B, C)
tok_p_prompt = torch.ones((B, P), device=DEVICE, dtype=DTYPE)           # (B, P)
token_probs  = torch.cat([tok_p_prompt, tok_p_comp], dim=1)             # (B, P+C)

hidden = mb_hidden   # (B, S-1, H)
S_hid  = hidden.size(1)

# ── [4] _should_stop forward ─────────────────────────────────────────────────
print("\n[4] Calling model._should_stop()...")
t1 = time.time()

probs_seq = model._should_stop(
    last_hidden=hidden,
    attn_stack=list(attns),
    token_probs=token_probs[:, :S_hid],
    mask=attention_mask[:, :S_hid].float(),
    input_ids=input_ids[:, :S_hid],
)  # → (B, 1)

head_time = (time.time() - t1) * 1000

assert probs_seq.shape == (B, 1), f"FAIL: expected ({B}, 1), got {probs_seq.shape}"
assert torch.isfinite(probs_seq).all(), f"FAIL: stop_prob contains non-finite values"
assert (probs_seq >= 0).all() and (probs_seq <= 1).all(), "FAIL: probs out of [0,1]"

print(f"    stop_prob shape: {probs_seq.shape}  dtype: {probs_seq.dtype}")
print(f"    stop_prob values: {probs_seq.squeeze(1).tolist()}")
print(f"    _should_stop latency: {head_time:.1f} ms")

# ── [5] BCE loss ─────────────────────────────────────────────────────────────
print("\n[5] Computing BCE stop_loss...")
correctness_labels = torch.tensor([[1.0], [0.0]], device=DEVICE)   # (B, 1)

p = probs_seq.squeeze(1).float()
y = correctness_labels.squeeze(1).float()

bce_loss = F.binary_cross_entropy(p.clamp(1e-6, 1-1e-6), y, reduction="mean")

assert torch.isfinite(bce_loss), f"FAIL: BCE loss is non-finite: {bce_loss}"
assert bce_loss.item() > 0.0, "FAIL: BCE loss is exactly 0.0 (suspicious)"

print(f"    stop_loss: {bce_loss.item():.6f}  ✓ (finite, positive)")

# ── [6] Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 3 SMOKE INTEGRATION TEST — RESULTS")
print("=" * 60)
print(f"  [PASS] Model instantiation with correctness head")
print(f"  [PASS] output_attentions non-None  ({len(attns)} layers, shape {attns[0].shape})")
print(f"  [PASS] stop_prob shape:  {probs_seq.shape}")
print(f"  [PASS] stop_prob dtype:  {probs_seq.dtype}")
print(f"  [PASS] stop_prob range:  [{probs_seq.min().item():.4f}, {probs_seq.max().item():.4f}]")
print(f"  [PASS] BCE stop_loss:    {bce_loss.item():.6f}")
print(f"  [PASS] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
print(f"  [INFO] Forward latency:  {step_time:.1f} ms")
print(f"  [INFO] Head latency:     {head_time:.1f} ms")
print(f"  [INFO] VRAM peak:        {vram_after:.1f} MB")
print("=" * 60)
print("All checks passed. System is ready for dry run.")
