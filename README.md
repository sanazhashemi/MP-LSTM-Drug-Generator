# MP-LSTM Drug Generator

This project implements a Multi-Path LSTM architecture to generate novel drug-like molecules in SMILES format using deep learning.

## 🔬 Description
The model is trained on a dataset of known molecules and is capable of generating novel, valid SMILES strings. The generated molecules are suitable for further analysis using tools such as SwissADME or similar.

## 🧠 Architecture
- Multi-Path LSTM (MP-LSTM)
- Trained with categorical cross-entropy loss
- Implemented in TensorFlow/Keras

## 🚀 How to Use

### 1. Clone the repository

```bash
git clone https://github.com/sanazhashemi/MP-LSTM-Drug-Generator.git
cd MP-LSTM-Drug-Generator
```

### 2. Install dependencies

Python 3.10 is recommended.

```bash
python3.10 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Generate molecules

The required input file (`valid_smiles.csv`) and trained model weights are included in the repository.

Run:

```bash
python generate.py
```

The generated molecules will be saved as:

```text
generated_smiles_with_MP-LSTM.csv
```

## 📄 Publication

This repository contains the implementation of the MP-LSTM model described in the following paper:

**Generating Pharmaceutical Molecules Using Multi-Path Deep Learning**

Sanaz Hashemipour, Habib Izadkhah, and Abolfazl Barzegar.  
DOI: https://doi.org/10.1109/IICAI70155.2026.11620965
  
## 📄 Citation
If you use this code or the MP-LSTM model in your research, please cite:
```bibtex
@inproceedings{11620965,
  author={Hashemipour, Sanaz and Izadkhah, Habib and Barzegar, Abolfazl},
  booktitle={2026 International Interdisciplinary Conference on Artificial Intelligence: Engineering, Health, Finance and Humanities (IICAI)},
  title={Generating Pharmaceutical Molecules Using Multi-Path Deep Learning},
  year={2026},
  pages={1-6},
  doi={10.1109/IICAI70155.2026.11620965}
}
```
---

## 📬 Contact
[Sanaz Hashemipour](mailto:sanazhashemipour2021@gmail.com)
