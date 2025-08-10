import cv2
import time
import numpy as np

def detect_features(cap, resolutions):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    results = {}
    for width, height in resolutions:
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if (actual_width, actual_height) != (width, height):
                print(f"Usando resolução {actual_width}x{actual_height} (solicitado {width}x{height})")
                width, height = actual_width, actual_height

            print(f"\nTestando em {width}x{height}...")
            
            total_time = 0
            frame_count = 0
            detected_cases = {
                'frontal': False,
                'lateral': False,
                'acessorios': False,
                'barba': False,
                'etnias': False
            }

            while frame_count < 100:
                ret, frame = cap.read()
                if not ret:
                    continue

                start_time = time.time()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detecta rostos
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.3,
                    minNeighbors=5,
                    minSize=(40, 40)
                )

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    
                    # Região de interesse para olhos e boca
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w]

                    eyes = eye_cascade.detectMultiScale(
                        roi_gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(15, 15)
                    )
                    
                    eye_positions = []
                    for (ex, ey, ew, eh) in eyes:
                        eye_center = x + ex + ew/2
                        if len(eye_positions) < 2: 
                            eye_positions.append((eye_center, (ex, ey, ew, eh)))

                    eye_positions.sort()
                    
                    for i, (center, (ex, ey, ew, eh)) in enumerate(eye_positions):
                        if i == 0:
                            color = (0, 255, 0)
                            label = "left eye"
                        else:
                            color = (0, 255, 0)
                            label = "right eye"
                        
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
                        cv2.putText(roi_color, label, (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Detecta boca (na metade inferior do rosto)
                    mouth_roi_gray = roi_gray[int(h/2):h, 0:w]
                    mouth_roi_color = roi_color[int(h/2):h, 0:w]
                    
                    mouths = mouth_cascade.detectMultiScale(
                        mouth_roi_gray,
                        scaleFactor=1.7,
                        minNeighbors=20,
                        minSize=(20, 15)
                    )
                    
                    for (mx, my, mw, mh) in mouths:
                        cv2.rectangle(mouth_roi_color, (mx, my), (mx+mw, my+mh), (255, 255, 0), 2)
                        cv2.putText(mouth_roi_color, "mouth", (mx, my-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                processing_time = time.time() - start_time
                total_time += processing_time
                frame_count += 1
                
                cv2.imshow(f'Detecção - {width}x{height} (Pressione Q para sair)', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if frame_count > 0:
                avg_time = total_time / frame_count
                results[f"{width}x{height}"] = avg_time
                print(f"Tempo médio: {avg_time:.4f}s por frame")

        except Exception as e:
            print(f"Erro na resolução {width}x{height}: {str(e)}")
            continue

    return results

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return

    resolutions = [
        (320, 240),
        (640, 480),
        (848, 600),
        (960, 540),
        (1280, 720)
    ]

    print("=== Iniciando detecção de rostos, olhos e boca ===")
    print("Instruções:")
    print("- Posicione o rosto frontalmente para detecção")
    print("- Vire o rosto para testar detecção lateral")
    print("- Use acessórios como óculos, chapéus, etc.")
    print("- Mostre diferentes estilos faciais (barba, sem barba)")
    print("- Pressione Q para sair quando terminar\n")

    results = detect_features(cap, resolutions)

    # Libera recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Resolução\tTempo Médio (s)")
    for res, time in results.items():
        print(f"{res}\t{time:.4f}")

if __name__ == "__main__":
    main()