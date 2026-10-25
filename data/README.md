# Data

The evaluation dataset is not stored in this repository. It is downloaded from Zenodo
([doi:10.5281/zenodo.17255307](https://doi.org/10.5281/zenodo.17255307), CC-BY 4.0) by

```bash
python code/00_download_data.py
```

which writes the six published CSV files to `data/zenodo_17255307/` and verifies each against its MD5
checksum. `code/03_clean_dataset.py` then writes the cleaned files to `data/processed/`, and
`notebooks/TwinGuard_experiments.ipynb` adds the graph windows `data/processed/graph_pipeline.npz`.

## Simulator data

`data/sim/` (dataset topology), `data/sim_nsfnet/`, and `data/sim_two_tier/` hold the calibration windows
(`cal.npz`), the feature normalization (`norm.npz`), and, for the other topologies, the traffic scale
(`topology.json`) of the simulator, which the closed-loop scripts and `code/verify.py` need. The training and
validation windows are not versioned; `code/06_sim_data.py` recreates them deterministically (about 1 min per
topology; `TG_TOPOLOGY=nsfnet` or `two_tier` for the other topologies).
