import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import psutil
import time
from model import build_mp_lstm  

filename = "valid_smiles.csv"
names = ['smiles', 'name']
data_df = pd.read_csv(filename, names=names)
smile = '\n'.join(data_df['smiles'].tolist())

unique_characters = sorted(list(set(smile)))
char_to_int = dict((c, i) for i, c in enumerate(unique_characters))
int_to_char = dict((i, c) for i, c in enumerate(unique_characters))

sequence_length = 10
X = []
Y = []

for i in range(len(smile) - sequence_length):
   seq_in = smile[i:i+sequence_length]
   seq_out = smile[i+sequence_length]
   X.append([char_to_int[char] for char in seq_in])
   Y.append(char_to_int[seq_out])

X_new = np.reshape(X, (len(X), sequence_length, 1)) / float(len(unique_characters))
y = to_categorical(Y)

input_shape = (X_new.shape[1], X_new.shape[2])
output_dim = y.shape[1]
model = build_mp_lstm(input_shape, output_dim)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint_path = "best-weights-8-1.23.keras"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min')

process = psutil.Process()

start_time = time.time()

history = model.fit(X_new, y, epochs=10, batch_size=64, callbacks=[checkpoint])

end_time = time.time()

memory_usage = process.memory_info().rss / 1024 / 1024

print(f"⏱️ Training time: {end_time - start_time:.2f} seconds")
print(f"💾 Memory usage: {memory_usage:.2f} MB") 

model.load_weights(checkpoint_path)
model.compile(loss='categorical_crossentropy', optimizer='adam')
