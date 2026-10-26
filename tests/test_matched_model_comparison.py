"""Regression checks for common inputs, frozen thresholds and training-only tuning."""
import importlib.util
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

PATH = Path(__file__).resolve().parents[1] / "scripts/run_matched_model_comparison.py"
SPEC = importlib.util.spec_from_file_location("matched_comparison", PATH)
matched = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matched)


def sample():
    rng = np.random.default_rng(728)
    X = rng.normal(size=(30, 4))
    y = np.repeat([0, 1], 15)
    X[:, 0] += y * .7
    X[:, 3] -= y
    return X, y


def test_inner_tuning_fits_preprocessing_only_on_inner_training(monkeypatch):
    X, y = sample()
    original = matched.fit_transform
    seen = []

    def recording_fit(values, labels):
        seen.append(values.copy())
        return original(values, labels)

    monkeypatch.setattr(matched, "fit_transform", recording_fit)
    fitted = matched.fit_models(X, y, seed=71)
    assert [len(v) for v in seen] == [20, 20, 20, 30]
    for values, split in zip(seen[:3], fitted["selection"]["splits"]):
        np.testing.assert_array_equal(values, X[split["train"]])
        assert not set(split["train"]) & set(split["validation"])


def test_frozen_predictions_and_thresholds_do_not_depend_on_test_companions():
    X, y = sample()
    fitted = matched.fit_models(X, y)
    thresholds = fitted["thresholds"].copy()
    mean, scale, signs = fitted["scaler"].mean_.copy(), fitted["scaler"].scale_.copy(), fitted["signs"].copy()
    z_one, _ = matched.target_coordinates(X[:1], fitted, "source_frozen")
    z_batch, _ = matched.target_coordinates(np.vstack([X[:1], np.ones((1, 4)) * 1e6]), fitted, "source_frozen")
    one = matched.predict_coordinates(z_one, fitted["models"])
    batch = matched.predict_coordinates(z_batch, fitted["models"])
    for method in matched.METHODS:
        np.testing.assert_allclose(one[method][0], batch[method][0], atol=1e-10)
    assert fitted["thresholds"] == thresholds
    np.testing.assert_array_equal(fitted["scaler"].mean_, mean)
    np.testing.assert_array_equal(fitted["scaler"].scale_, scale)
    np.testing.assert_array_equal(fitted["signs"], signs)


def test_direction_alignment_is_the_same_coordinate_change_for_learned_models():
    X, y = sample()
    fitted = matched.fit_models(X, y)
    prior = np.array([-1., 1., 1., 1.])
    target = X[:8] * [3., 2., 1., .5] + 4.
    Z, state = matched.target_coordinates(target, fitted, "target_aligned", prior)
    source_z = fitted["scaler"].transform(X)
    raw_model = LogisticRegression(C=fitted["parameters"]["logistic"], solver="lbfgs", max_iter=2000).fit(source_z, y)
    # Sign-coding training inputs and target inputs must be algebraically equivalent
    # to applying the corresponding diagonal alignment in raw standardized coordinates.
    raw_target_z = Z * fitted["signs"]
    expected = raw_model.decision_function(raw_target_z)
    actual = fitted["models"]["logistic"].decision_function(Z)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)
    assert state["signs"] == prior.tolist()
    np.testing.assert_allclose(matched.predict_coordinates(Z, fitted["models"])["equal_mean"], Z.mean(axis=1))
    np.testing.assert_allclose(matched.predict_coordinates(Z, fitted["models"])["equal_mean_c15"],
                               np.clip(Z, -1.5, 1.5).mean(axis=1))
