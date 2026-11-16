# Ubuntu/Linux and VS Code installation guide

## 1. Extract and install
Use Python 3.11 or 3.12 (the recorded execution used 3.12). From the directory containing the ZIP:

```bash
unzip Energy_Recommender_System_Complete.zip
cd energy-recommender-system
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential graphviz
python3 --version
bash scripts/setup.sh
source .venv/bin/activate
python -m pip check
```

`setup.sh` creates `.venv`, installs requirements, and installs the local package. The NumPy version is intentionally pinned for the tested Surprise build. The compiler is GCC by default. If your system Python is outside 3.10–3.12, create the virtual environment explicitly with an installed Python 3.12 (`python3.12 -m venv .venv`) before installation. Python 3.11/3.12 are the recommended choices. No API keys or cloud services are needed.

## 2. Run the verified workflow

```bash
python -m pytest -q
python -m src.pipeline --config config.yaml
python tools/generate_diagrams.py
python tools/generate_result_figures.py
python scripts/recommend.py --model results/models/ImplicitALS_11.joblib --household H0000 --daytype 0 --k 5
```

The pipeline generates its own labelled synthetic readings unless `--input` is supplied. It fits eight models for each of three seeds, runs counterfactual simulations, ablations, sensitivity and uncertainty calculations. It writes results to `results/` and processed inputs to `data/processed/`. Existing outputs at these locations are overwritten. Model inference returns compatibility rankings; the constrained hybrid recommendations and factor explanations are in `results/tables/recommendations.csv`.

## 3. VS Code
Open the extracted project directory using **File → Open Folder**, or run `code .` if the VS Code CLI is installed. Install Microsoft's Python and Jupyter extensions. Use **Python: Select Interpreter** to choose `.venv/bin/python`. Open the terminal in the project root and run the commands above. For notebooks, use **Select Kernel** and choose the same environment.

```bash
python -m pip install -r requirements-notebooks.txt
python tools/execute_notebooks.py
```

The five notebooks include executed outputs in the ZIP. Execution regenerates them with in-process IPython in the installed environment (no kernel sockets required). Read them in sequence from dataset exploration to result visualisation. Graphviz's `dot` command is needed for diagram regeneration, but existing figure files can be viewed without it.

## 4. Local dashboard

```bash
streamlit run dashboard.py
```

Open the localhost URL printed in the terminal. Select a household, seed and weekday/weekend context to inspect history, assumed tariff, constrained recommendations, explanations, model metrics and simulated outcomes. Stop the server with Ctrl+C. This is a local results viewer, not a live smart-meter controller.

## 5. Run on original LCL data
Download the original full or split CSV archive from the official provider linked in README and extract it manually. No real meter files are included in this package.

```bash
mkdir -p data/raw/lcl
# Extract your downloaded CSV archive into data/raw/lcl/.
python scripts/preprocess.py --input data/raw/lcl --config config.yaml --output data/processed_lcl_preview
python -m src.pipeline --config config.yaml --input data/raw/lcl
```

Before running: decide the source timestamp convention; confirm kWh units; set `max_households` to a justified memory-bounded cohort; and preserve original tariff-group metadata outside the three-column loader. The retained first-encountered cohort is not random or representative. The code does not ingest the historical tariff spreadsheet; it applies the declared simulated tariff scenario even to real supplied meter data.

For naive timestamps, set `timezone` to their documented source timezone. For UK local naive clocks crossing DST, supply unambiguous offsets after checking source metadata. Do not blindly assign UTC to local clocks. The loader rejects ambiguous clocks, missing timestamps, invalid energy and conflicting duplicates. It never fills unknown intervals with zero. All local slots and dates are defined for Europe/London. The full experiment intentionally rejects data that cannot support evaluable warm/cold contexts or at least one complete 48-slot test day, with an actionable error; standalone preprocessing can still audit a sparse file.

## 6. Troubleshooting
- **Compiler missing:** install `build-essential` and `python3-dev`, then rerun `CC=gcc python -m pip install -r requirements.txt` inside the virtual environment.
- **NumPy/Surprise binary mismatch:** use a fresh isolated `.venv`; do not mix user-site packages or upgrade NumPy independently of the tested pins.
- **No evaluable contexts:** obtain more complete days/weekends or revise the declared affinity threshold as a new experiment. Never adjust thresholds based on final test performance for the reported experiment.
- **Not enough households:** the combined benchmark requires at least two households and a nonzero held-out cold fraction below one. For single-household inspection use the loader/preprocessing functions instead.
- **High memory use:** reduce `max_households`. Chunked CSV reading does not make retained data or dense models out-of-core.
- **Different results:** compare `config.yaml`, package versions, input hash, Python version and seeds. Small numerical differences may arise across BLAS/platform implementations.

See `results/logs/`, `results/metadata.json` and `docs/execution_report.md` for the actual delivered-run evidence. The recorded environment freeze is supplementary; install the portable top-level requirements rather than blindly restoring machine-specific packages.
