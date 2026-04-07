# 📄 NGCF — Neural Graph Collaborative Filtering
### Research Paper Replication

> **Paper:** *Neural Graph Collaborative Filtering*
> Wang, X., He, X., Wang, M., Feng, F., & Chua, T.-S. (SIGIR 2019)
> [`Paper 10.pdf`](document/Paper%2010.pdf)

---

## 🧠 Overview

This repository contains a faithful **PyTorch reproduction** of **NGCF (Neural Graph Collaborative Filtering)**, a graph-based recommendation model proposed by Wang et al. at SIGIR 2019.

NGCF explicitly encodes collaborative signals in user-item interaction graphs using graph neural network propagation. The model builds a bipartite user–item graph, constructs a normalized Laplacian, and performs multi-layer embedding propagation — capturing high-order connectivity for better recommendations.

This replication was carried out as part of a **Recommender Systems research project** and includes:
- Full PyTorch training pipeline with BPR loss
- Faithful implementation of all equations from the paper
- Evaluation on three benchmark datasets (Gowalla, Yelp2018, Amazon-Book)
- Publication-grade result visualizations
- Pre-training with Matrix Factorization embeddings (optional)

---

## 📁 Project Structure

```
REC_SYS_PRO/
│
├── src/                            # Source code
│   ├── main_rec.py                 # Main training & evaluation script
│   ├── config_rec.py               # Hyperparameter configuration (argparse)
│   ├── data_loader_rec.py          # Dataset loading + BPR negative sampling
│   ├── adjacency_rec.py            # Normalized Laplacian construction (Eq. 8)
│   ├── metrics_rec.py              # Recall@K and NDCG@K evaluation
│   ├── early_stopping_rec.py       # Early stopping (patience=50)
│   ├── pretrain_mf_rec.py          # Optional MF pre-training
│   ├── GENERATION.ipynb            # Publication-grade figure generation notebook
│   ├── TEST_rec.ipynb              # Full test & analysis notebook
│   ├── TEST2_rec.ipynb             # Ablation study notebook
│   └── TEST3_rec.ipynb             # Additional experiments notebook
│
├── data/
│   └── data/
│       ├── gowalla/                # Gowalla check-in dataset
│       │   ├── train.txt
│       │   └── test.txt
│       ├── yelp2018/               # Yelp 2018 dataset
│       │   ├── train.txt
│       │   └── test.txt
│       └── amazon-book/            # Amazon-Book dataset
│           ├── train.txt
│           └── test.txt
│
├── results/
│   └── outputs/
│       └── paper_package/          # All generated figures (PNG)
│           ├── fig1_bipartite_graph.png
│           ├── fig2_architecture.png
│           ├── fig5_dropout.png
│           ├── fig6_training_curves.png
│           ├── table1_statistics.png
│           ├── table2_performance.png
│           ├── table3_layer_ablation.png
│           ├── table4_variants.png
│           └── novel_*.png         # Novel analysis figures
│
├── document/
│   └── Paper 10.pdf                # Original NGCF paper (Wang et al., SIGIR 2019)
│
└── ngcf_requirements_and_pip_commands.txt   # Dependency list and install commands
```

---

## 📐 Key Paper Concepts Implemented

| Component | Paper Reference | File |
|---|---|---|
| Bipartite graph Laplacian | Section 2.2.2, Eq. 8 | `adjacency_rec.py` |
| Embedding propagation layers | Section 2.2, Eq. 3–7 | NGCF model |
| BPR pairwise loss | Section 2.4.1, Eq. 11 | `main_rec.py` |
| Message dropout | Section 2.4.2 | model forward pass |
| Node dropout | Section 2.4.2 | `adjacency_rec.py` |
| Full-ranking evaluation | Section 4.2.1 | `metrics_rec.py` |
| Early stopping (patience=50) | Section 4.2.3 | `early_stopping_rec.py` |
| MF pre-training | Section 4.2.3, Footnote 4 | `pretrain_mf_rec.py` |

---

## 📊 Datasets

Three publicly available datasets are used, following the paper's setup (Section 4.1):

| Dataset | #Users | #Items | #Interactions | Density |
|---|---|---|---|---|
| Gowalla | 29,858 | 40,981 | 1,027,370 | 0.00084 |
| Yelp2018 | 31,668 | 38,048 | 1,561,406 | 0.00130 |
| Amazon-Book | 52,643 | 91,599 | 2,984,108 | 0.00062 |

Each file follows the format:
```
user_id  item_id_1  item_id_2  ...
```
An 80/20 train/test split is pre-applied. Training interactions build the interaction matrix **R** used to construct the graph Laplacian.

---

## ⚙️ Setup & Installation

### 1. Clone / Extract the project

```bash
unzip REC_SYS_PRO.zip
cd REC_SYS_PRO
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate       # Linux / Mac
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

**One-command install:**
```bash
pip install --upgrade pip
pip install numpy scipy torch matplotlib seaborn pandas scikit-learn jupyterlab notebook
```

**Or install individually:**
```bash
pip install numpy
pip install scipy
pip install torch
pip install matplotlib
pip install seaborn
pip install pandas
pip install scikit-learn
pip install jupyterlab
```

> **Note on PyTorch:** Installation may vary by hardware. For Apple Silicon (M1/M2), `pip install torch` is recommended. For CUDA-enabled GPU training, see [pytorch.org](https://pytorch.org/get-started/locally/).

---

## 🚀 Running the Code

### Train NGCF

```bash
cd src/

python main_rec.py \
    --dataset gowalla \
    --data_path ../data/data/ \
    --embed_size 64 \
    --n_layers 3 \
    --layer_sizes 64,64,64 \
    --lr 0.0001 \
    --batch_size 1024 \
    --epoch 400 \
    --reg 1e-5 \
    --mess_dropout 0.1 \
    --node_dropout 0.0 \
    --Ks 20 \
    --patience 50 \
    --save_path ../results/outputs/ \
    --device auto
```

Switch datasets with `--dataset yelp2018` or `--dataset amazon-book`.

### Optional: Pre-train MF Embeddings

```bash
python pretrain_mf_rec.py --dataset gowalla --data_path ../data/data/

# Then use them in NGCF training:
python main_rec.py --dataset gowalla --pretrain ../results/outputs/mf_gowalla_pretrain.pth ...
```

---

## 🔧 Hyperparameters

All default values match the paper (Section 4.2.3):

| Argument | Default | Description |
|---|---|---|
| `--embed_size` | 64 | Embedding dimensionality |
| `--n_layers` | 3 | Number of propagation layers |
| `--layer_sizes` | `64,64,64` | Size per layer (must match `n_layers`) |
| `--lr` | 0.0001 | Adam learning rate |
| `--batch_size` | 1024 | BPR training batch size |
| `--epoch` | 400 | Maximum training epochs |
| `--reg` | 1e-5 | L2 regularization coefficient (λ) |
| `--mess_dropout` | 0.1 | Message dropout rate |
| `--node_dropout` | 0.0 | Node dropout rate |
| `--Ks` | 20 | Evaluation cut-off K (comma-separated) |
| `--patience` | 50 | Early stopping patience (epochs) |
| `--seed` | 2019 | Random seed |
| `--eval_interval` | 5 | Evaluate every N epochs |
| `--device` | auto | `auto`, `cpu`, `cuda`, or `mps` |

---

## 📈 Results

### Recall@20 — Replication vs. Paper (Table 2)

| Dataset | Paper (Recall@20) | Replicated (Recall@20) |
|---|---|---|
| Gowalla | 0.1569 | 0.1378 |
| Yelp2018 | 0.0579 | 0.0522 |

### Layer Depth Ablation — Recall@20 (Table 3)

| Layers | Paper Gowalla | Replicated Gowalla | Paper Yelp | Replicated Yelp |
|---|---|---|---|---|
| NGCF-1 | 0.1556 | 0.1233 | 0.0543 | 0.0482 |
| NGCF-2 | 0.1547 | 0.1282 | 0.0566 | 0.0499 |
| NGCF-3 | 0.1569 | 0.1378 | 0.0579 | 0.0522 |
| NGCF-4 | 0.1570 | 0.1373 | 0.0566 | 0.0517 |

> The replication gap is attributed to hardware differences, training duration constraints, and the absence of full MF pre-training in some runs. Trends and relative rankings across layers are faithfully reproduced.

---

## 🖼️ Generated Figures

All figures are saved in `results/outputs/paper_package/`. They include:

| Figure | Description |
|---|---|
| `fig1_bipartite_graph.png` | User-item bipartite graph visualization |
| `fig2_architecture.png` | NGCF model architecture diagram |
| `fig5_dropout.png` | Effect of dropout on performance |
| `fig6_training_curves.png` | Loss and recall training curves |
| `table1_statistics.png` | Dataset statistics table |
| `table2_performance.png` | Full performance comparison (Table 2) |
| `table2_recall_bars.png` | Recall@20 bar chart by method |
| `table2_ndcg_bars.png` | NDCG@20 bar chart by method |
| `table2_heatmap.png` | Performance heatmap across datasets |
| `table3_layer_ablation.png` | Layer depth ablation results |
| `table4_variants.png` | Propagation variant comparison (Table 4) |
| `novel_a_gap_dashboard.png` | Paper vs. replication performance gap |
| `novel_b_convergence.png` | Convergence analysis |
| `novel_c_mf_improvement.png` | MF pre-training impact |
| `novel_d_radar.png` | Multi-metric radar chart |
| `novel_e_param_efficiency.png` | Parameter efficiency plot |
| `novel_f_embedding_breakdown.png` | Embedding dimension analysis |
| `novel_g_training_time.png` | Training time per epoch |
| `novel_h_summary_dashboard.png` | Overall replication summary dashboard |

To regenerate all figures, open and run `src/GENERATION.ipynb` in JupyterLab.

---

## 📓 Notebooks

| Notebook | Purpose |
|---|---|
| `GENERATION.ipynb` | Generates all paper-package figures |
| `TEST_rec.ipynb` | Full training run, logging, and analysis |
| `TEST2_rec.ipynb` | Layer ablation experiments |
| `TEST3_rec.ipynb` | Additional experiments and comparisons |

---

## 📚 Reference

```bibtex
@inproceedings{wang2019neural,
  title     = {Neural Graph Collaborative Filtering},
  author    = {Wang, Xiang and He, Xiangnan and Wang, Meng and Feng, Fuli and Chua, Tat-Seng},
  booktitle = {Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages     = {165--174},
  year      = {2019}
}
```

---

## 📋 Dependencies Summary

| Package | Purpose |
|---|---|
| `torch` | Model training, GPU/MPS support |
| `numpy` | Numerical operations |
| `scipy` | Sparse matrix (Laplacian) construction |
| `matplotlib` | Figure generation |
| `seaborn` | Statistical visualizations |
| `pandas` | Data handling for analysis |
| `scikit-learn` | Utilities (e.g., t-SNE) |
| `jupyterlab` | Running `.ipynb` notebooks |

---

*Replication project — NGCF (Wang et al., SIGIR 2019)*
