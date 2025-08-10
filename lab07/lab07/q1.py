import cv2
import time

import cv2.data

def get_safe_resolutions():
    return [
        (320, 240),
        (640, 480),
        (848, 600),
        (960, 540),
        (1280, 720)
    ]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

safe_resolutions = get_safe_resolutions()
execution_times = []

for width, height in safe_resolutions:
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if (actual_width, actual_height) != (width, height):
            print(f"Aviso: Resolução {width}x{height} não suportada. Usando {actual_width}x{actual_height}")
            width, height = actual_width, actual_height
        
        total_time = 0
        frame_count = 0
        print(f"\nTestando em {width}x{height}...")
        
        while frame_count < 100:
            ret, frame = cap.read()
            if not ret:
                print("Aviso: Frame não capturado. Tentando continuar...")
                continue
            
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                start = time.time()
                
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=6,
                    minSize=(40, 40)
                )
                
                total_time += time.time() - start
                frame_count += 1
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                cv2.imshow('Face Detection - Pressione Q para sair', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print(f"Erro no processamento do frame: {str(e)}")
                continue
        
        if frame_count > 0:
            avg_time = total_time / frame_count
            execution_times.append((width, height, avg_time))
            print(f"Sucesso! Média: {avg_time:.4f}s por frame")
            
    except Exception as e:
        print(f"Falha na resolução {width}x{height}: {str(e)}")
        continue

cap.release()
cv2.destroyAllWindows()
print("Resolução\tTempo Médio (s)")
for res in execution_times:
    print(f"{res[0]}x{res[1]}\t{res[2]:.4f}")