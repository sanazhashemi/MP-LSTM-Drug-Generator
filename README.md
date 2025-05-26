# MP-LSTM Drug Generator

This project implements a Multi-Path LSTM architecture to generate novel drug-like molecules in SMILES format using deep learning.

## 🔬 Description
The model is trained on a dataset of known molecules and is capable of generating novel, valid SMILES strings. The generated molecules are suitable for further analysis using tools such as SwissADME or similar.

## 🧠 Architecture
- Multi-Path LSTM (MP-LSTM)
- Trained with categorical cross-entropy loss
- Implemented in TensorFlow/Keras

## 🚀 How to Use

1. Clone the repository:

   git clone https://github.com/sanazhashemi/MP-LSTM-Drug-Generator.git
   cd MP-LSTM-Drug-Generator
   

2. Create a Python virtual environment (Python 3.8 or higher recommended) and install dependencies:
    
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    
    
3. Run the molecule generation script:
    python generate.py

  
## 📄 Citation
This repository is associated with a research article currently under peer review. Citation information will be provided here upon publication.

---

## 📬 Contact
[Sanaz Hashemipour](mailto:sanazhashemipour2021@gmail.com)
