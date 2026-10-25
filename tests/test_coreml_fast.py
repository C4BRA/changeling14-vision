import pytest
import torch

from changeling14_vision.model.encoder import build_encoder
from changeling14_vision.replays.detect import DetectModel
from changeling14_vision.replays.predict import guesses, guesses_fast


def _model():
    torch.manual_seed(0)
    m = DetectModel(build_encoder("nano"), text_dim=16).eval()
    m.text_vectors = torch.randn(50, 16)
    m.something.bias.data.fill_(2.0)            # most squares pass, so every slot path is exercised
    m.more.bias.data.fill_(0.0)
    return m


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_fast_guesses_match_guesses(threshold):
    m = _model()
    names = [f"thing {i}" for i in range(50)]
    x = torch.rand(1, 3, 96, 128)
    with torch.no_grad():
        out = m(x)
    a = guesses(m, names, x, threshold, out=out)[0]
    b = guesses_fast(m, names, out, threshold)
    assert len(a) == len(b) and len(a) > 0
    for ga, gb in zip(a, b):
        assert (ga.name, ga.mob, ga.slot) == (gb.name, gb.mob, gb.slot)
        assert [n for n, _ in ga.names] == [n for n, _ in gb.names]
        assert max(abs(u - v) for u, v in zip(ga.box, gb.box)) < 1e-3
        assert abs(ga.score - gb.score) < 1e-5 and abs(ga.player - gb.player) < 1e-5


def test_export_wrapper_matches_the_model():
    from changeling14_vision.replays import coreml
    m = _model()
    ref = DetectModel(build_encoder("nano"), text_dim=16).eval()
    ref.load_state_dict(m.state_dict())
    w = coreml._Export(m, 96, 128).eval()
    x = torch.rand(1, 3, 96, 128)
    with torch.no_grad():
        want, got = ref(x), w(x)
    for key, g in zip(coreml.OUTPUTS, got):
        assert torch.allclose(want[key].reshape(g.shape), g, atol=1e-5), key
