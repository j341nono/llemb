# llmembed

Unified embedding extraction from Decoder-only LLMs.

## Features

- **Backends**: Support for Hugging Face Transformers and VLLM.
- **Pooling Strategies**:
    - `mean`: Average pooling of all tokens (excluding padding).
    - `last_token`: Vector of the last token.
    - `eos_token`: Vector corresponding to the EOS token position.
    - `prompt_eol`: Embeddings extracted using a prompt template targeting the last token.
- **Quantization**: Support for 4-bit and 8-bit quantization via `bitsandbytes`.
- **Layer Selection**: Extract embeddings from any layer (default: last hidden state).

## Installation

Install using `uv`:

```bash
uv pip install llmembed
```

To include VLLM support:

```bash
uv pip install llmembed[vllm]
```

To include quantization support:

```bash
uv pip install llmembed[quantization]
```

## Usage

Initialize the encoder and extract embeddings:

```python
import llmembed

# Initialize encoder with specific backend and configuration
enc = llmembed.Encoder(
    model_name="sshleifer/tiny-gpt2",
    backend="transformers",
    device="cpu",
    quantization=None  # Options: "4bit", "8bit", or None
)

# Extract embeddings using mean pooling
embeddings = enc.encode("Hello world", pooling="mean")
print(embeddings.shape)
```

## Development

Clone the repository and sync dependencies:

```bash
git clone https://github.com/j341nono/llmembed.git
cd llmembed
uv sync --all-extras --dev
```

Run tests:

```bash
uv run pytest
```

Run static analysis:

```bash
uv run ruff check src
uv run mypy src
```
