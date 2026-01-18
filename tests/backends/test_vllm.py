import sys
from unittest.mock import MagicMock, patch

# Mock vllm BEFORE importing the backend to ensure it's available for the test logic
mock_vllm = MagicMock()
sys.modules["vllm"] = mock_vllm

# Now import
from llmembed.backends.vllm_backend import VLLMBackend

def test_vllm_init():
    # Setup mock LLM in the backend module if needed, 
    # but since we mocked vllm module, LLM in backend is already our mock.
    
    # We need to verify VLLMBackend calls LLM(...)
    with patch("llmembed.backends.vllm_backend.LLM") as mock_llm_cls:
        backend = VLLMBackend("model-name")
        mock_llm_cls.assert_called_once()

def test_vllm_encode():
    with patch("llmembed.backends.vllm_backend.LLM") as mock_llm_cls:
        backend = VLLMBackend("model-name")
        
        # Mock the encode output
        mock_output = MagicMock()
        # Assume output has outputs.embedding
        mock_output.outputs.embedding = [0.1, 0.2]
        
        # Configure instance
        backend.model.encode.return_value = [mock_output]
        
        res = backend.encode("test")
        assert res.shape == (1, 2)
        assert res[0][0] == 0.1
