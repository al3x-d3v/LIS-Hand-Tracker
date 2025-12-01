import pandas as pd
import numpy as np
import pickle # Serve per salvare il modello
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. CARICAMENTO DATI
# Assicurati che il nome del file sia quello giusto (es. 'command_data.csv')
print("⏳ Caricamento dataset...")
data = pd.read_csv('command_data.csv', header=None)

# 2. SEPARAZIONE INPUT (X) e OUTPUT (y)
# La colonna 0 contiene le tue lettere ('a', 'b', ...) -> È la nostra y
# Le colonne dalla 1 alla fine contengono i numeri -> È la nostra X
X = data.iloc[:, 1:].values  # Prendi tutte le righe, colonne da 1 in poi
y_letters = data.iloc[:, 0].values # Prendi tutte le righe, solo colonna 0

print(f"Dati caricati. Trovati {len(data)} campioni.")
print(f"Classi trovate (lettere): {np.unique(y_letters)}")

# 3. TRADUZIONE (Label Encoding)
# Qui trasformiamo le lettere in numeri per l'AI
encoder = LabelEncoder()
y = encoder.fit_transform(y_letters)

# Stampiamo la legenda per ricordarcela
print("\nLegenda creata automaticamente:")
for id_num, letter in enumerate(encoder.classes_):
    print(f"   {id_num} --> {letter}")

# Salva l'encoder! Ci servirà nel main.py per ritradurre i numeri in lettere
f = open('label_encoder.p', 'wb')
pickle.dump(encoder, f)
f.close()

# 4. DIVISIONE TRAIN / TEST
# Usiamo il 20% dei dati per testare se il modello funziona
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

# 5. ADDESTRAMENTO (Il cervello)
print("\nAvvio addestramento modello (Random Forest)...")
model = RandomForestClassifier()
model.fit(x_train, y_train)

# 6. VALUTAZIONE
# Chiediamo al modello di predire i dati di test che non ha mai visto
score = model.score(x_test, y_test)
print(f"Addestramento completato!")
print(f"Precisione del modello: {score * 100:.2f}%")

# 7. SALVATAGGIO MODELLO
# Salviamo il cervello in un file statico
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
print("💾 Modello salvato in 'model.p'")