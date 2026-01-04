# Spectrum-Based-Generative-EEG-Modeling

This repository contains an end-to-end pipeline for **generating EEG signals via time–frequency (spectrogram) representations** using **generative models** (GAN variants and diffusion-style generation, as implemented in this repo). The core idea is to convert EEG into a spectrogram representation, train a generative model in that space, and reconstruct realistic EEG signals back in the time domain.

The project is structured as a **reproducible experiment pipeline** driven by **Hydra configuration**, with clear separation between:
- configuration (`config/`)
- experiment entrypoints (`src/`)
- reusable pipeline/model logic (`lib/`)


## What’s Included

### Time–Frequency EEG Generation
- Spectrogram-based EEG representation for model training
- Reconstruction back to time-domain EEG via inverse transforms (as implemented in the pipeline)

### Generative Models (as implemented in `lib/gen/models/`)
This repo includes multiple generator/discriminator implementations and training setups, including:
- WGAN-style models and conditional variants
- Architectures organized via a unified model factory (`lib/gen/models/_utils.py`)

### Experiment & Pipeline System (Hydra)
- Central config entry: `config/config.yaml`
- Experiment presets: `config/experiment/`
- Pipeline definitions: `config/pipeline/`
- Dataset presets: `config/dataset/`

## Quickstart

### 1) Create the environment

**Option A — Conda (recommended, matches repo setup)**
```bash
conda env create -f environment.yml
conda activate env
```
**Option B – pip**
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Run an experiment (Hydra)
The main entrypoint is:
`python src/main.py`

To run the generation pipeline preset, use the provided config groups:
`python src/main.py experiment=gen pipeline=gen dataset=airplane_ccwgan`


#### Notes:
The available dataset configs are in `config/dataset/` (e.g., `airplane_ccwgan.yaml`, `airplane_ccdcwgan.yaml`).
The main generation logic is wired through `src/gen/_run_gen.py` and `lib/experiment/pipeline/gen.py`.


## Repository Structure

```text
.
├── config/
│   ├── config.yaml
│   ├── dataset/
│   │   ├── airplane_ccwgan.yaml
│   │   └── airplane_ccdcwgan.yaml
│   ├── experiment/
│   │   ├── base.yaml
│   │   └── gen.yaml
│   ├── gen/
│   │   └── gen.yaml
│   └── pipeline/
│       ├── base.yaml
│       └── gen.yaml
│
├── lib/
│   ├── experiment/
│   │   └── pipeline/
│   │       ├── base.py
│   │       └── gen.py
│   ├── gen/
│   │   ├── models/
│   │   ├── trainer/
│   │   └── ...
│   └── ...
│
├── src/
│   ├── main.py
│   ├── base/
│   │   └── _run_base.py
│   └── gen/
│       ├── run.py
│       └── _run_gen.py
│
├── environment.yml
├── requirements.txt
└── README.md
```


## How to Add Your Own Dataset / Experiment
#### Add a new dataset preset
1. Create a new YAML under `config/dataset/` (copy one of the existing presets).
2. Point it to your input paths / preprocessing parameters expected by the pipeline.
Run:
`python src/main.py experiment=gen pipeline=gen dataset=<your_dataset_yaml_name_without_ext>`

#### Add a new model
1. Implement the model under `lib/gen/models/`.
2. Register it through the model factory in:
`lib/gen/models/_utils.py`

## Outputs
This project uses Hydra, so outputs (logs, checkpoints, artifacts) are typically written under Hydra’s run directory (depending on your Hydra settings). If you want a fixed output path, add/override Hydra’s `hydra.run.dir` in your config or CLI override.
