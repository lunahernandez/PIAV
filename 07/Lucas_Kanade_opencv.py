import cv2
import numpy as np

VIDEO_PATH = "07/videos/people2.mp4"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"No se pudo abrir el vídeo: {VIDEO_PATH}")
        return

    # Parámetros para detectar esquinas (buenos puntos a seguir)
    feature_params = dict(
        maxCorners=500,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    # Parámetros de Lucas-Kanade
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # Primer frame
    ret, old_frame = cap.read()
    if not ret:
        print("No se pudo leer el primer frame.")
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculamos flujo óptico (Lucas-Kanade)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is None:
            break

        # seleccionamos puntos válidos
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Dibujamos las trayectorias
        for (new, old) in zip(good_new, good_old):
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            mask = cv2.line(mask, (int(x_new), int(y_new)), (int(x_old), int(y_old)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(x_new), int(y_new)), 3, (0, 0, 255), -1)

        img = cv2.add(frame, mask)

        cv2.imshow("Lucas-Kanade (sparse)", img)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break

        # Actualizamos
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
