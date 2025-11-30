import cv2
import mediapipe as mp
import data_processing
import save_date

# 1. Configurazione MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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

    k =  cv2.waitKey(2)
    # 5. VISUALIZZAZIONE
    # Se abbiamo trovato delle mani...
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:           
            lista=data_processing.pre_pocessing_landmark(hand_landmarks.landmark)
            
            if k & 0xFF == ord('a'):        
                save_date.save_on_csv('a', lista)
            elif k & 0xFF == ord('b'):
                save_date.save_on_csv('b', lista)
            elif k & 0xFF == ord('c'):
                save_date.save_on_csv('c', lista)
            elif k & 0xFF == ord('d'):
                save_date.save_on_csv('d', lista)
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

