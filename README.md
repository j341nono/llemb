# llmembed

Unified embedding extraction from Decoder-only LLMs.

## Installation

```bash
uv pip install llmembed
# or
pip install llmembed
```

To support VLLM backend:
```bash
uv pip install llmembed[vllm]
```

To support Quantization (bitsandbytes):
```bash
uv pip install llmembed[quantization]
```

## Usage

```python
import llmembed

# Initialize encoder
enc = llmembed.Encoder(
    model="sshleifer/tiny-gpt2",
    backend="transformers",
    device="cpu", # or "cuda"
    quantization=None # or "bitsandbytes"
)

# Extract embeddings
embeddings = enc.encode("Hello world", pooling="mean")
print(embeddings.shape)
```

## Features

- **Backends**: Transformers, VLLM.
- **Pooling**: `mean`, `last_token`, `eos_token`, `prompt_eol`.
- **Quantization**: 4-bit/8-bit via bitsandbytes.