import pytest
from llmembed.core import Encoder
from llmembed.backends.transformers_backend import TransformersBackend

def test_encoder_init_transformers():
    enc = Encoder(model="sshleifer/tiny-gpt2", backend="transformers")
    assert isinstance(enc.backend_instance, TransformersBackend)

def test_encoder_invalid_backend():
    with pytest.raises(ValueError):
        Encoder(model="sshleifer/tiny-gpt2", backend="invalid")

def test_encoder_encode():
    enc = Encoder(model="sshleifer/tiny-gpt2", backend="transformers")
    res = enc.encode("hello")
    assert res is not None
