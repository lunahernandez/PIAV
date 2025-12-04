import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- CONFIGURACIÓN ---
VIDEO_PATH = "videos/people2.mp4"
model = YOLO('yolo11x-pose.pt')

# Parámetros de Lucas-Kanade
lk_params = dict(
    winSize=(25, 25),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Inicializar video
cap = cv2.VideoCapture(VIDEO_PATH)

# Variables de estado
p0 = None           # El punto actual que estamos siguiendo
old_gray = None     # El frame anterior en escala de grises
mask = None         # Para dibujar la estela

# --- NUEVO: Lista para guardar el historial de puntos ---
history_points = [] 

# --- BUCLE PRINCIPAL ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -----------------------------------------------------------
    # FASE 1: INICIALIZACIÓN CON YOLO
    # -----------------------------------------------------------
    if p0 is None:
    
        # Ejecutar seguimiento YOLOv8 en el fotograma, persistiendo los rastreos entre fotogramas
        results = model.track(frame)[0]

        for r in results:
            kpts = r.keypoints
            nk = kpts.shape[1]
            for i in range(nk):
                if i==10: # muñeca derecha
                    keypoint=kpts.xy[0,i]    
                    p0 = np.array([[int(keypoint[0]), int(keypoint[1])]], dtype=np.float32).reshape(-1, 1, 2)

    # -----------------------------------------------------------
    # FASE 2: SEGUIMIENTO CON LUCAS-KANADE
    # -----------------------------------------------------------
    else:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None and st[0] == 1:
            new_x, new_y = p1[0].ravel()
            old_x, old_y = p0[0].ravel()

            # Dibujar trazo
            mask = cv2.line(mask, (int(new_x), int(new_y)), (int(old_x), int(old_y)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(new_x), int(new_y)), 5, (0, 0, 255), -1)

            old_gray = frame_gray.copy()
            p0 = p1.reshape(-1, 1, 2)
            
            # --- GUARDAR PUNTO EN HISTORIAL ---
            history_points.append([new_x, new_y])
            
        else:
            print("Punto perdido. Reiniciando YOLO...")
            p0 = None
            mask = np.zeros_like(frame)

    # Mostrar resultado
    if mask is not None:
        img = cv2.add(frame, mask)
    else:
        img = frame

    cv2.imshow("Tracking + Gráfica", img)

    key = cv2.waitKey(30) & 0xFF
    if key == 27: # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()

# -----------------------------------------------------------
# FASE 3: GENERACIÓN DE GRÁFICAS (Tu código)
# -----------------------------------------------------------
if len(history_points) > 0:
    print(f"Generando gráficas con {len(history_points)} puntos registrados...")
    
    t = np.linspace(0, len(history_points), num=len(history_points))
    
    x_points = []
    y_points = []
    
    for point in history_points:
        x_points.append(int(point[0]))
        y_points.append(int(point[1]))

    plt.figure(1, (12,6))
    
    # Gráfica coordenada X
    plt.subplot(1,2,1)
    plt.plot(t, x_points, color='blue')
    plt.ylabel("Coordenada X")
    plt.xlabel("Tiempo (Frames)")
    plt.title("Coordenada X vs Tiempo")
    plt.grid(True)
    
    # Gráfica coordenada Y
    plt.subplot(1,2,2)
    plt.plot(t, y_points, color='orange')
    plt.ylabel("Coordenada Y") # Nota: En imagen (0,0) es arriba-izquierda, por lo que Y aumenta hacia abajo
    plt.xlabel("Tiempo (Frames)")
    plt.title("Coordenada Y vs Tiempo")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
else:
    print("No se registraron puntos para graficar.")