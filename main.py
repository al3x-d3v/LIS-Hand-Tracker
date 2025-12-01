import cv2
import mediapipe as mp
import numpy as np
import data_processing
import pickle
import save_date

# 1. Configurazione MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#Caricamento modello
print("Carico moello AI")
try :
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    encoder = pickle.load(open('./label_encoder.p', 'rb'))
    print("Modello Caricato correttamente!")
except Exception as e:
    print("Errore nel caricamento del modello ai : {e}" )
    exit()
# Inizializziamo il modello
# static_image_mode=False perché è un video, non una foto
# max_num_hands=1 per ora, così è più veloce
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# 2. Configurazione Webcam
# Usa l'indice 0 per la webcam di default
cap = cv2.VideoCapture(1)

print("Avvio webcam... Premi '!' per uscire.")

while cap.isOpened():
    # Leggiamo il frame dalla webcam
    success, image = cap.read()
    if not success:
        print("Non riesco a leggere dalla webcam")
        continue

    # 3. PRE-PROCESSING
    # OpenCV ci dà l'immagine in BGR, ma MediaPipe la vuole in RGB.
    # Cerca la costante di conversione colore BGR -> RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    # 4. INFERENZA (Il cuore dell'AI)
    # Passiamo l'immagine RGB al modello
    results = hands.process(image_rgb)

    k =  cv2.waitKey(100)
    cv2.rectangle(image, (0,0), (160, 60), (0,0,0), -1)
    # 5. VISUALIZZAZIONE
    # Se abbiamo trovato delle mani...
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:           
            lista=data_processing.pre_pocessing_landmark(hand_landmarks.landmark)
            # Codice riconoscimento
            probabilities = model.predict_proba([lista])[0]
            best_class_id = np.argmax(probabilities)
            highest_prob = np.max(probabilities)
            threshold = 0.9
            if highest_prob>threshold :
                predicted_character = encoder.inverse_transform([best_class_id])[0]
                #print(f"Lettera : {predicted_character}")
                cv2.putText(image, 
                        predicted_character, 
                        (20, 45), # Posizione (x, y)
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5,      # Grandezza Font
                        (255, 255, 255), # Colore Bianco
                        3,        # Spessore
                        cv2.LINE_AA)
            #Codice ADDESTRAMENTO
            '''
            if k & 0xFF == ord('a'):        
                save_date.save_on_csv('a', lista)
            elif k & 0xFF == ord('b'):
                save_date.save_on_csv('b', lista)
            elif k & 0xFF == ord('c'):
                save_date.save_on_csv('c', lista)
            elif k & 0xFF == ord('d'):
                save_date.save_on_csv('d', lista)
            '''

            
            # Disegniamo lo scheletro sull'immagine ORIGINALE (non quella RGB)
            # mp_hands.HAND_CONNECTIONS serve per disegnare le linee tra i punti
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS)

    # Mostriamo l'immagine a schermo in una finestra chiamata "LIS Hand Tracker"
    cv2.imshow('LIS Hand Tracker', image)

    # 6. USCITAq
    # Aspetta 5ms e controlla se è stato premuto il tasto 'q' (codice ASCII 113)
    if k & 0xFF == ord('!'):
        break
    
# Pulizia finale
cap.release()
cv2.destroyAllWindows()

