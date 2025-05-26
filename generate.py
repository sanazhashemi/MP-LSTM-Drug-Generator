import numpy as np
import pandas as pd
from rdkit import Chem
from model import build_mp_lstm

filename = "/Users/sanaz/Documents/paper/codes/Rev1-Com3/valid_smiles.csv"
names = ['smiles', 'name']
data_df = pd.read_csv(filename, names=names)
smile = '\n'.join(data_df['smiles'].tolist())

unique_characters = sorted(list(set(smile)))
number_vocabulary = len(unique_characters)
char_to_int = dict((c, i) for i, c in enumerate(unique_characters))
int_to_char = dict((i, c) for i, c in enumerate(unique_characters))

sequence_length = 10
X = []

for i in range(len(smile) - sequence_length):
    seq_in = smile[i:i+sequence_length]
    X.append([char_to_int[char] for char in seq_in])

# بارگذاری مدل و وزن‌ها
input_shape = (sequence_length, 1)
output_dim = number_vocabulary

model = build_mp_lstm(input_shape, output_dim)
model.load_weights("best-weights-8-1.23.keras")
model.compile(loss='categorical_crossentropy', optimizer='adam')

# انتخاب تصادفی یک دنباله اولیه
startIndex = np.random.randint(0, len(X) - 1)
Seq = X[startIndex].copy()

def sample_with_temperature(preds, temperature=0.8):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

generated_smiles = set()
current_seq = Seq.copy()
generated = ""
all_smiles = []

for _ in range(5000):
    seq = np.reshape(current_seq, (1, len(current_seq), 1))
    seq = seq / float(number_vocabulary)
    
    generation = model.predict(seq, verbose=0)
    newIndex = sample_with_temperature(generation[0], temperature=0.8)
    
    next_char = int_to_char[newIndex]
    generated += next_char
    current_seq.append(newIndex)
    current_seq = current_seq[1:]
    
    if next_char == '\n':
        mol = Chem.MolFromSmiles(generated.strip())
        is_valid = mol is not None and generated.strip() not in generated_smiles
        if is_valid:
            print(f"\n✅ مولکول معتبر: {generated.strip()}")
            generated_smiles.add(generated.strip())
        else:
            print(f"\n❌ مولکول نامعتبر: {generated.strip()}")
        all_smiles.append({'smiles': generated.strip(), 'is_valid': is_valid})
        generated = ""

df = pd.DataFrame(all_smiles)
df.to_csv('generated_smiles_with_MP-LSTM.csv', index=False)
print("خروجی‌ها در فایل 'generated_smiles_with_MP-LSTM.csv' ذخیره شدند.")
