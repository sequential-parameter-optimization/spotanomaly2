"""Estimator + ForecasterRecursive factory.

Maps a model-name string from config to a fresh sklearn-compatible estimator
(``_build_estimator``) and wraps it in a configured ``ForecasterRecursive``
(``_create_forecaster``). Estimators are imported lazily inside the dispatch so
optional dependencies (lightgbm / xgboost / catboost) don't load unless their
model is requested.

``_filter_params`` drops kwargs that the chosen estimator's ``__init__`` does
not accept, so leftover entries in a per-channel YAML don't blow up the build.
"""

import inspect
from typing import Any

from .estimators import (
    _NEEDS_SCALING,
    KernelRidgeApprox,
    SVRApprox,
    _CatBoostRegressor,
    _MLPRegressorRobust,
)


def _filter_params(estimator_cls, raw_params: dict[str, Any], logger=None, model_name: str = "") -> dict[str, Any]:
    """Drop kwargs that ``estimator_cls.__init__`` does not accept."""
    sig = inspect.signature(estimator_cls.__init__)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw:
        return raw_params
    valid_keys = {key for key in sig.parameters if key != "self"}
    filtered = {k: v for k, v in raw_params.items() if k in valid_keys}
    dropped = sorted(k for k in raw_params if k not in valid_keys)
    if dropped and logger is not None:
        logger.warning(f"Ignoring unsupported params for {model_name}: {dropped}")
    return filtered


def _build_estimator(
    model_name: str,
    model_params: dict[str, Any] | None,
    random_seed: int,
    logger=None,
):
    """Map a model name to a fresh, unfitted sklearn-compatible estimator."""
    name = (model_name or "").strip().lower()
    params = dict(model_params or {})

    def _filt(cls):
        return _filter_params(cls, params, logger=logger, model_name=model_name)

    if name in {"lightgbm", "lgbm", "lgbmregressor"}:
        from lightgbm import LGBMRegressor

        params.setdefault("random_state", random_seed)
        params.setdefault("verbose", -1)
        return LGBMRegressor(**_filt(LGBMRegressor))

    if name in {"xgboost", "xgb", "xgbregressor"}:
        from xgboost import XGBRegressor

        params.setdefault("random_state", random_seed)
        return XGBRegressor(**_filt(XGBRegressor))

    if name in {"catboost", "catboostregressor"}:
        if _CatBoostRegressor is None:
            raise ImportError("catboost is not installed")
        params.setdefault("random_seed", random_seed)
        params.setdefault("verbose", 0)
        # CatBoost otherwise writes a ``catboost_info/`` log dir (+ ``tmp/``) into
        # the process cwd on every fit — litter on a server, and one folder per
        # tuning trial. Nothing here reads those logs; disable by default. A user
        # can re-enable via config (allow_writing_files / train_dir pass through).
        params.setdefault("allow_writing_files", False)
        return _CatBoostRegressor(**_filt(_CatBoostRegressor))

    if name in {"ridge", "ridgeregressor"}:
        from sklearn.linear_model import Ridge

        return Ridge(**_filt(Ridge))

    if name in {"elasticnet", "elasticnetregressor"}:
        from sklearn.linear_model import ElasticNet

        params.setdefault("random_state", random_seed)
        return ElasticNet(**_filt(ElasticNet))

    if name in {"lasso", "lassoregressor"}:
        from sklearn.linear_model import Lasso

        params.setdefault("random_state", random_seed)
        return Lasso(**_filt(Lasso))

    if name in {"bayesianridge", "bayesian_ridge"}:
        from sklearn.linear_model import BayesianRidge

        return BayesianRidge(**_filt(BayesianRidge))

    if name in {"huber", "huberregressor"}:
        from sklearn.linear_model import HuberRegressor

        return HuberRegressor(**_filt(HuberRegressor))

    if name in {"kernelridgeapprox", "kernel_ridge_approx", "nystroemkernelridge"}:
        params.setdefault("random_state", random_seed)
        return KernelRidgeApprox(**_filt(KernelRidgeApprox))

    if name in {"svrapprox", "svr_approx", "nystroemsvr"}:
        params.setdefault("random_state", random_seed)
        return SVRApprox(**_filt(SVRApprox))

    if name in {"mlp", "mlpregressor"}:
        params.setdefault("random_state", random_seed)
        params.setdefault("max_iter", 500)
        params.setdefault("early_stopping", True)
        return _MLPRegressorRobust(**_filt(_MLPRegressorRobust))

    raise ValueError(
        f"Unsupported model '{model_name}'. Supported: LightGBM, XGBoost, CatBoost, "
        "Ridge, ElasticNet, Lasso, BayesianRidge, Huber, KernelRidgeApprox, SVRApprox, MLP"
    )


def _create_forecaster(
    model_name: str,
    model_params: dict[str, Any] | None,
    n_lags: int | list[int],
    *,
    random_seed: int = 42,
    has_exog: bool = False,
    logger=None,
):
    """Create a ``ForecasterRecursive`` configured for ``model_name``.

    For scale-sensitive models (linear / kernel / MLP) we pass spotforecast2's
    ``transformer_y`` (and ``transformer_exog`` when exog is present) so that
    the library handles standardisation itself. This avoids wrapping the
    estimator at the sklearn level — which previously collided with
    ``ForecasterRecursive._recursive_predict``'s LinearModel fast path and
    produced NaN cascades.
    """
    from sklearn.preprocessing import StandardScaler
    from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

    estimator = _build_estimator(model_name, model_params, random_seed=random_seed, logger=logger)

    needs_scaling = (model_name or "").strip().lower() in _NEEDS_SCALING
    transformer_y = StandardScaler() if needs_scaling else None
    transformer_exog = StandardScaler() if (needs_scaling and has_exog) else None

    return ForecasterRecursive(
        estimator=estimator,
        lags=n_lags,
        transformer_y=transformer_y,
        transformer_exog=transformer_exog,
    )
