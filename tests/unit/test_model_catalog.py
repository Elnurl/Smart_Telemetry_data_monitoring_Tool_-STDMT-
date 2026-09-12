from app.models.model_types import (
    TOUCHED_44_MODEL_NAMES,
    catalog_display_name,
    format_model_catalog_label,
    resolve_runtime_model_type,
    to_internal_model_type,
)


def test_catalog_label_marks_aliases_and_keeps_native_names():
    assert format_model_catalog_label("Isolation Forest") == "Isolation Forest"
    assert format_model_catalog_label("TimesFM") == "TimesFM  (uses Prophet)"
    assert format_model_catalog_label("LSTM-AE") == "LSTM-AE  (uses LSTM)"


def test_catalog_display_name_strips_uses_suffix():
    assert catalog_display_name("TimesFM  (uses Prophet)") == "TimesFM"
    assert to_internal_model_type("TimesFM  (uses Prophet)") == "timesfm"
    assert resolve_runtime_model_type(to_internal_model_type("ARIMA")) == "prophet"


def test_every_catalog_name_resolves_to_a_backend():
    for name in TOUCHED_44_MODEL_NAMES:
        runtime = resolve_runtime_model_type(to_internal_model_type(name))
        assert runtime
        label = format_model_catalog_label(name)
        assert catalog_display_name(label) == name
