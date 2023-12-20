import cv2
from emotion_detection_test import cnn_emotion_predictor

# Classificador de rosto pré-treinado
cascade_rosto = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Captura de vídeo da webcam 
captura = cv2.VideoCapture(0)

while True:
    # Capturar o próximo frame 
    ret, frame = captura.read()

    # Converter o frame para escala de cinza
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos no frame
    rosto = cascade_rosto.detectMultiScale(frame_cinza, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Desenhar retângulos ao redor dos rostos detectados
    for (x, y, w, h) in rosto:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
        # Extrair os pixels dentro do retângulo do rosto
        face_roi = frame[y:y+h, x:x+w]

        # Pré-processamento
        face_resized = cv2.resize(face_roi, (48, 48))
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        face_normalized = face_gray / 255.0  # Normalize to [0, 1]

        predictor = Cnn_emotion_predictor()
        predictor.predict(face_normalized)
    


    # Mostrar o frame com os retângulos desenhados
    cv2.imshow('Detecção de Rosto na Webcam', frame)

    # Verificar se a tecla 'p' foi pressionada para encerrar o loop
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

# Liberar os recursos e fechar as janelas
captura.release()
cv2.destroyAllWindows()
