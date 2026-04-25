import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

# Use a tiny config for memory/speed, but preserve the Llama-3.1-8B structure.
config = LlamaConfig(
    vocab_size=128256,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=2,  # just 2 layers to save memory for test
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=131072,
    _attn_implementation="eager" # Force eager to get output_attentions=True
)

model = LlamaForCausalLM(config).eval()

batch_size = 2
seq_len = 16
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
attention_mask = torch.ones((batch_size, seq_len))

print("=== Task 1: Compatibility Audit Runtime Test ===")
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True
    )

print(f"last_hidden_state shape: {outputs.hidden_states[-1].shape}")
print(f"attentions shape (per layer): {outputs.attentions[0].shape}, total layers returned: {len(outputs.attentions)}")
print(f"logits shape: {outputs.logits.shape}")
