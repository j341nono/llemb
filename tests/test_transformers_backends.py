import llmembed


def test_encode_with_transformers_backend():
    enc = llmembed.Encoder(
        model="sshleifer/tiny-gpt2",
        backend="transformers",
        device="cpu",
    )
    vec = enc.encode("hello")

    assert vec.ndim == 1
    assert vec.shape[0] > 0
