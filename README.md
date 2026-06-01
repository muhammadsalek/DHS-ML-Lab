
### Module Explanations

| Module | Responsibility |
|--------|----------------|
| `data/` | Streaming, windowing, normalization, train/val/test split |
| `models/` | All forecasting architectures + sktime wrappers |
| `xai/` | Post-hoc and intrinsic explainability modules |
| `evaluation/` | Metrics (sMAPE, MASE, CRPS), backtests, residual analysis |
| `pipeline.py` | Single entry point for full reproducibility |

---

## 🔬 Core Machine Learning / Research Pipeline

ChronosX follows a strict, auditable ML pipeline suitable for academic and industrial research:

| Step | Description |
|------|-------------|
| **Data Collection** | Supports `.csv`, `.parquet`, InfluxDB, and simulated data |
| **Cleaning & Preprocessing** | Interpolation, anomaly capping, frequency alignment |
| **Feature Engineering** | Rolling stats, fourier terms, lag features, categorical encoding |
| **Model Development** | Hyperparameter search via Optuna, cross-validation |
| **Model Evaluation** | sMAPE, MASE, QLIKE, Diebold-Mariano tests |
| **Explainability (XAI)** | Global + local feature importance, temporal attention |
| **Reproducibility Pipeline** | Fixed random seeds, Docker + Conda, MLflow logging |

---

## 🤖 Machine Learning Models

### Classical ML Models

| Model | Use Case | Strengths | Limitations |
|-------|----------|-----------|--------------|
| **XGBoost** | Tabular forecasting with lags | Fast, handles missing data, robust | No temporal inductive bias |
| **LightGBM** | Large-scale, many features | GOSS + EFB, lower memory | Requires careful tuning |
| **ARIMA/SARIMA** | Baseline univariate | Interpretable, small data | Fails on high-frequency, multivariate |

### Advanced ML Models

| Model | Use Case | Strengths | Limitations |
|-------|----------|-----------|--------------|
| **Temporal Fusion Transformer (TFT)** | Interpretable forecasting | Attention + static covariates + quantiles | Computationally heavy |
| **Informer** | Very long sequences | ProbSparse attention, low complexity | Less interpretable than TFT |
| **N-BEATS** | Pure DL, no covariates | SOTA on M4, interpretable via stacks/blocks | Univariate only |
| **DeepAR** | Probabilistic forecasting | Likelihood-based, good for sparse data | Needs many time series |

### Deep Learning Models (PyTorch)

| Model | Type | Input Shape | Output |
|-------|------|-------------|--------|
| **Transformer** | Encoder-decoder | (batch, seq_len, n_features) | (batch, pred_len, n_targets) |
| **LSTM + Attention** | RNN with context vector | (batch, seq_len, n_features) | (batch, pred_len, n_targets) |
| **TCN** | Causal dilated conv | (batch, seq_len, n_features) | (batch, pred_len, n_targets) |

---

## 🔍 Explainable AI (XAI)

Interpretability is not an afterthought in ChronosX — it is enforced at every evaluation step.

| Method | Scope | Output |
|--------|-------|--------|
| **SHAP (KernelExplainer)** | Global + local | Feature importance summary, waterfall plots |
| **LIME (TabularExplainer)** | Local (single sample) | Linear surrogate explanations |
| **Temporal Attention Maps** | Model-specific | Attention weights over input time steps |
| **Integrated Gradients** | Deep models only | Per-feature, per-timestep attribution |

**Why interpretability matters:**  
In finance, medicine, and infrastructure monitoring, a black-box forecast is insufficient. Regulators and domain experts require *why* a prediction was made. ChronosX provides attribution at both the feature level and time-step level, enabling trust and debugging.

---

## 📚 Research Papers & Related Work

### Foundational Papers

- **Temporal Fusion Transformer** – Lim, B., et al. (2021). *Temporal Fusion Transformers for interpretable multi-horizon time series forecasting.* International Journal of Forecasting.
- **Informer** – Zhou, H., et al. (2021). *Informer: Beyond efficient transformer for long sequence time-series forecasting.* AAAI.
- **N-BEATS** – Oreshkin, B. N., et al. (2020). *N-BEATS: Neural basis expansion analysis for interpretable time series forecasting.* ICLR.
- **DeepAR** – Salinas, D., et al. (2020). *DeepAR: Probabilistic forecasting with autoregressive recurrent networks.* International Journal of Forecasting.

### Modern Research (2022–2025)

- **PatchTST** – Nie, Y., et al. (2023). *A time series is worth 64 words: Long-term forecasting with transformers.* ICLR.
- **TimeX** – Zheng, X., et al. (2024). *Learning temporal representations with learnable positional encoding.* NeurIPS TS Workshop.
- **SHAP for Time Series** – Bento, J., et al. (2022). *TimeSHAP: Explaining recurrent models through sequence perturbations.* XAI in Time Series workshop.

### Related Repositories

- `tsai` – Fastai-based time series deep learning
- `darts` – User-friendly forecasting library
- `sktime` – Unified API for classical + ML models

ChronosX distinguishes itself via **tight integration of TFT, SHAP, and reproducibility-first design** — not just a model zoo.

---

## 📂 Dataset Section

### Primary Supported Datasets

| Dataset | Domain | Samples | Freq | Target Variables |
|---------|--------|---------|------|------------------|
| **ETT (Electricity Transformer)** | Energy | 2 years | 15min/1h | Oil temperature |
| **Electricity (UCI)** | Energy | 321 clients | 1h | Consumption (kWh) |
| **Traffic (PeMS)** | Transportation | 862 sensors | 1h | Occupancy rate |
| **Weather (Jena)** | Climate | ~10 years | 10min | Temperature, pressure, etc. |

### Dataset Structure (after preprocessing)

```python
Dataset(
    x = torch.Tensor(n_samples, lookback, n_features),   # Input
    y = torch.Tensor(n_samples, horizon, n_targets),     # Target
    times = pd.DatetimeIndex(n_samples),                 # For seasonality
    static = torch.Tensor(n_samples, n_static)          # Optional
)
