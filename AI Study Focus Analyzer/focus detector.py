import cv2
import time
import math

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

camera = cv2.VideoCapture(0)

focused_time = 0.0
distracted_time = 0.0

last_face_position = None
last_timestamp = time.time()

movement_threshold = 45

while True:
    success, frame = camera.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    state = "Distracted"

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_center = (x + w // 2, y + h // 2)

        face_region = gray[y:y + h, x:x + w]
        eyes = eye_detector.detectMultiScale(face_region, 1.1, 4)

        face_stable = False
        eyes_open = len(eyes) > 0

        if last_face_position is not None:
            dx = face_center[0] - last_face_position[0]
            dy = face_center[1] - last_face_position[1]
            movement = math.sqrt(dx * dx + dy * dy)

            if movement < movement_threshold:
                face_stable = True
        else:
            face_stable = True

        if face_stable and eyes_open:
            state = "Focused"

        last_face_position = face_center

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame,
                          (x + ex, y + ey),
                          (x + ex + ew, y + ey + eh),
                          (0, 255, 0), 2)
    else:
        last_face_position = None

    current_time = time.time()
    elapsed_time = current_time - last_timestamp

    if state == "Focused":
        focused_time += elapsed_time
    else:
        distracted_time += elapsed_time

    last_timestamp = current_time

    cv2.putText(frame, "Status: " + state, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, "Focused Time: " + str(int(focused_time)) + " seconds",
                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, "Distracted Time: " + str(int(distracted_time)) + " seconds",
                (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("AI Study Focus Analyzer", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()
