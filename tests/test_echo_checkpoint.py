import numpy as np

from echo_module.models import final_echo_model


def test_saved_echo_checkpoint_loads_without_imagenet_download(monkeypatch):
    original_mobilenet = final_echo_model.MobileNetV2

    def assert_no_imagenet_download(*args, **kwargs):
        assert kwargs["weights"] is None
        return original_mobilenet(*args, **kwargs)

    monkeypatch.setattr(final_echo_model, "MobileNetV2", assert_no_imagenet_download)
    model = final_echo_model.build_echo_model(
        weights_path="echo_module/models/echo_best_model.keras"
    )

    prediction = model(np.zeros((1, 224, 224, 3), dtype=np.float32), training=False).numpy()

    assert prediction.shape == (1, 4)
    assert np.isfinite(prediction).all()
    assert np.isclose(prediction.sum(), 1.0, atol=1e-5)
