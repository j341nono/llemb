from typing import Any, List, Optional, Union

from .backends.transformers_backend import TransformersBackend
from .interfaces import Backend

# Try importing VLLMBackend, might fail if vllm is not installed or dependencies missing
try:
    from .backends.vllm_backend import VLLMBackend
except ImportError:
    VLLMBackend = None # type: ignore

class Encoder:
    def __init__(
        self,
        model: str,
        backend: str = "transformers",
        device: str = "cpu",
        quantization: Optional[str] = None
    ):
        """
        Initialize the Encoder.

        Args:
            model: Model identifier.
            backend: Backend to use ('transformers', 'vllm').
            device: Device ('cpu', 'cuda', etc.).
            quantization: Quantization config.
        """
        self.backend_name = backend
        self.backend_instance: Backend
        
        if backend == "transformers":
            self.backend_instance = TransformersBackend(model, device, quantization)
        elif backend == "vllm":
            # Check if VLLMBackend class is available
            if VLLMBackend is None:
                # Try importing again to see specific error or if it was just skipped
                try:
                    from .backends.vllm_backend import VLLMBackend as VBackend
                    self.backend_instance = VBackend(model, device, quantization)
                except ImportError as e:
                    raise ImportError(f"VLLM backend requires 'vllm' installed. Error: {e}")
            else:
                 self.backend_instance = VLLMBackend(model, device, quantization)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def encode(
        self,
        text: Union[str, List[str]],
        pooling: str = "mean",
        layer_index: int = -1,
        **kwargs: Any
    ) -> Any:
        """
        Encode text into embeddings.

        Args:
            text: Input text or list of texts.
            pooling: Pooling strategy ('mean', 'last_token', 'eos_token', 'prompt_eol').
            layer_index: Layer index to extract embeddings from.
            **kwargs: Backend specific arguments.

        Returns:
            Embeddings as numpy array or torch tensor.
        """
        return self.backend_instance.encode(
            text, pooling=pooling, layer_index=layer_index, **kwargs
        )
