# IDM-VTON with Body and Garment Measurements (Thesis)

This repository is a fork of [IDM-VTON](https://github.com/yisol/IDM-VTON) (diffusion-based virtual try-on built on Stable Diffusion XL). In this project, the IDM-VTON model is extended by body and garment measurements as additional conditioning signals, so that the generated try-on image reflects how a garment of given dimensions fits a person of given dimensions.

The original IDM-VTON documentation is in [`README.md`](README.md). This file only describes what was added.

## Measurement conditioning

Seven scalar measurements are used: `body_bust`, `body_height`, `body_hips`, `body_waist`, `garment_bust`, `garment_length`, `garment_sleeve_length`. Two derived ease features (`bust_ease`, `hem_drop`) are computed from them. A `MeasurementEncoder` turns the normalized values into an embedding, which is injected into TryonNet in one of three ways:

| Variant | Injection | Training script |
|---|---|---|
| Base extension | One measurement token appended to the cross-attention context (LoRA on attention layers) | `train_with_measurements.py` |
| FiLM extension | Feature-wise modulation (gamma, beta) of ResNet features in the deepest blocks | `train_with_film.py` |
| Measurement Adapter | Third, gated, parallel cross-attention branch with its own K/V per layer (IP-Adapter principle) | `train_with_measurement_adapter.py` |

Training and evaluation use the FIT dataset (person, cloth and target images plus `measurements.json`).

## Added files

Model components (`src/`)
- `measurement_encoder.py` - MLP (optional Fourier features) mapping measurements to an embedding
- `measurement_adapter.py` - Measurement Adapter with token projector, per-layer K/V, gates
- `film_generator.py` - FiLM generator heads per modulated ResNet block
- `fit_dataset.py` - `FITDatasetWithMeasurements` dataset class
- `counterfactual.py` - implementation for measurement controllability tests (shift one measurement by +/- sigma, everything else fixed)

Modified files
- `src/tryon_pipeline.py` - optional `measurement_tokens` argument
- `gradio_demo/utils_mask.py` - measurement-based bust and length expansion of the agnostic mask
- `gradio_demo/app.py` - loads the `MeasurementEncoder` and a checkpoint

Preprocessing and data
- `preprocess_fit.py` - computes agnostic masks, DensePose maps, ground-truth garment masks and BLIP-2 garment captions for FIT
- `data/sample_data.py` - samples a subset of the FIT parquet files
- `data/compute_measurement_stats.py` - mean and std of the measurements (used for normalization)

Training
- `train_with_measurements.py`, `train_with_film.py`, `train_with_measurement_adapter.py` - one per variant (see above)
- `train_hpo.py`, `train_hpo_film.py`, `train_hpo_measurement_adapter.py` - Optuna hyperparameter search for each variant

Inference
- `inference_fit.py` - baseline / token variant on FIT data
- `inference_fit_facerestore.py` - same, with face and hair pasted back from the input image
- `inference_film.py` - FiLM variant
- `inference_measurement_adapter.py` - Measurement Adapter variant
- All three accept `--counterfactual` to generate the +/- sigma sweeps

Evaluation and analysis
- `experiments/evaluate.py` - evaluation pipeline (image metrics, garment-mask comparison, counterfactual analysis via `--cf_dir`)
- `probe_measurement_sensitivity.py` - checks whether the measurement embedding changes the UNet output
- `empirical.py`, `bust_line_check.py` - empirical shoulder-hip distance measurement and visual check of the assumed bust line
- `test_expand_mask_new.py` - visual test of the bust and length mask expansion

Cloud jobs (`jobs/vertex/`)
- Vertex AI job specs for training, HPO, inference and the sensitivity probe, grouped by variant (`baseline/`, `film/`, `adapter/`, `misc/`)
- `Dockerfile`, `cloudbuild.yaml` - container used by the jobs

## Additional information

- Some files are not included: model checkpoints (`ckpt/`), datasets (`data/`) and training outputs (`result_fit/`, `tb_logs/`)
- The same weights as the original project (DensePose, human parsing, OpenPose, IP-Adapter, CLIP image encoder under `ckpt/`) are required, plus the pretrained `yisol/IDM-VTON` weights from Hugging Face
- License: CC BY-NC-SA 4.0 (non-commercial), inherited from IDM-VTON

## Use of AI tools

Claude (Anthropic) was used as a support tool for the implementation of this project.
