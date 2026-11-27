# BASGA

BASGA is a boundary-aware spatial graph autoencoder for spatial transcriptomics.

## Installation

Use Python 3.11.18 and the CUDA 12.1 build of PyTorch. For mclust-based clustering, use R 4.3.3 with mclust 6.1.1.

```bash
python -m pip install 'torch==2.4.1+cu121' --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-server.txt
```

## DLPFC 151676

Place the original filtered counts, author layer annotations and spatial coordinates in the following layout, or pass their parent directory with `--data-dir`. [Data reference](https://stagate.readthedocs.io/en/latest/T1_DLPFC.html)

```text
data/
├── 151676_filtered_feature_bc_matrix.h5
├── 151676_truth.txt
└── spatial/
    └── tissue_positions_list.csv
```

The annotation file has two tab-separated columns without a header, containing the barcode and its layer label. The coordinate file uses the original six-column Visium format without a header.

```bash
python run.py --data-dir ./data --check-only
python run.py --data-dir ./data --seed 10 --device cuda:0
```

The default run uses 3,000 highly variable genes, a radius of 150, a 512/30-dimensional autoencoder and 1,000 training epochs. mclust EEE produces seven clusters. ARI is evaluated on the 3,431 annotated spots among the 3,460 input spots.

Results and figures are written under `outputs/`. `latest_audit_seed10.json` gives the completed run location. Use `--plot-only --seed 10` to redraw saved results, or `--smoke --device cpu` for a three-epoch workflow check.

The graph attention backbone follows [STAGATE](https://doi.org/10.1038/s41467-022-29439-6). See [LICENSE_STAGATE.txt](LICENSE_STAGATE.txt) for the upstream license.
