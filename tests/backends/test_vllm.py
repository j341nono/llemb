from unittest.mock import MagicMock, patch

from llmembed.backends.vllm_backend import VLLMBackend


def test_vllm_init():
    # We need to verify VLLMBackend calls LLM(...)
    with patch("llmembed.backends.vllm_backend.LLM") as mock_llm_cls:
        _ = VLLMBackend("model-name")
        mock_llm_cls.assert_called_once()

def test_vllm_encode():
    with patch("llmembed.backends.vllm_backend.LLM") as mock_llm_cls:
        mock_instance = mock_llm_cls.return_value
        
        # Create a mock output object that mimics vLLM's RequestOutput structure for embed()
        mock_output = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1, 0.2]
        mock_output.outputs = [mock_embedding]
        
        # Configure instance to return this structure for embed()
        mock_instance.embed.return_value = [mock_output]
        
        backend = VLLMBackend("model-name")
        res = backend.encode("test")
        
        assert res.shape == (1, 2)
        assert res[0][0] == 0.1
        mock_instance.embed.assert_called_once()
