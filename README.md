# Misinformation Spread as a Graph Process: Network Topology, Structural Immunization, and Regime Dependence

This package contains the manuscript, Supporting Online Material, executed notebooks, source code, analysis data, statistical outputs, and figures for **Misinformation Spread as a Graph Process: Network Topology, Structural Immunization, and Regime Dependence**.

## Main deliverables

- `final_documents/MisinformationOnlineCitation.docx`
- `final_documents/MisinformationStandardCitation.docx`
- `final_documents/MisinformationSOM.docx`

## Executed notebooks

- `notebooks/baseline_analysis.ipynb`: baseline diffusion, direct-exposure outcomes, descriptive and inferential statistics.
- `notebooks/intervention_analysis.ipynb`: eight placement strategies, coverage response, repeated-measures tests, and mechanism outcomes.
- `notebooks/robustness_mechanisms_scalability.ipynb`: near-threshold sweeps, network-parameter sensitivity, seed placement, behavioral heterogeneity, survival tests, structural mechanisms, and runtime scaling.

All three notebooks were executed without errors before delivery.

## Reproducible pipeline

Run commands from the package root. Python 3.13.5 was used for the delivered results.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

python code/run_primary_experiments.py
python code/run_robustness_experiments.py
python code/analyze_primary.py
python code/analyze_robustness.py
python code/create_figures.py
```

The runner scripts default to the directory structure used during development. To rerun directly from this portable package, either update their output-path constants to `data/` and `analysis_outputs/`, or use the executed notebooks, which read the included final CSV files.

## Folder guide

- `documents/`: Word files.
- `notebooks/`: executed analyses.
- `code/`: simulation, experiment, analysis, and figure script
- `data/`: primary and robustness simulation records.
- `analysis_outputs/`: statistical summaries, pairwise comparisons, survival tests, strategy winners, mechanism correlations, and scalability summaries
- `figures/`: supplementary figures.
