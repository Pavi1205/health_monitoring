import cv2
import mediapipe as mp
import numpy as np
import time

# -----------------------
# Face Detection
# -----------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# For heart rate simulation
prev_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -----------------------
    # Detect face
    # -----------------------
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mp_draw.draw_landmarks(frame, faceLms, mp_face.FACEMESH_TESSELATION)


        # -----------------------
        # Body temperature estimation (simulate)
        # -----------------------
        red_channel = frame[:,:,2]
        avg_red = np.mean(red_channel)
        body_temp = 36 + (avg_red - 120)/50  # simulate around 36-38 C

        # -----------------------
        # Heart rate simulation (for BP)
        # -----------------------
        frame_count += 1
        curr_time = time.time()
        fps = frame_count / (curr_time - prev_time)
        heart_rate = int(fps*2)  # just simulation
        systolic = 100 + int(heart_rate/2)  # simulated BP
        diastolic = 70 + int(heart_rate/4)

        # -----------------------
        # Display metrics
       # -----------------------
        cv2.putText(frame, f"Body Temp: {body_temp:.1f} C", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(frame, f"Heart Rate: {heart_rate} bpm", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Blood Pressure: {systolic}/{diastolic} mmHg", (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    cv2.imshow("Health Monitor Simulation", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
