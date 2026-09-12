import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Hand Lanmarker initialise
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

landmarker = vision.HandLandmarker.create_from_options(options)


# Finger count

def count_fingers(landmarks):

    fingers = []

    # Thumb
    if landmarks[4].x < landmarks[3].x:
        fingers.append(True)
    else:
        fingers.append(False)

    # Index, Middle, Ring and Little finger
    for tip in [8, 12, 16, 20]:

        if landmarks[tip].y < landmarks[tip - 2].y:
            fingers.append(True)
        else:
            fingers.append(False)

    return fingers.count(True)

# Webcam 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not access camera")
    exit()

# Loop whether read frame

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Could not read frame")
        break

    # Mirror the camera
    frame = cv2.flip(frame, 1)

# Conver bgr to rgb

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

# convert mediapipe image

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

# Detect hand

    results = landmarker.detect(mp_image)

# If hand is detected

    if results.hand_landmarks:

        for hand_landmarks in results.hand_landmarks:

# count fingers

            count = count_fingers(hand_landmarks)

            # Display finger count
            cv2.putText(
                frame,
                f"Fingers: {count}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )
            #  GET FRAME SIZE

            h, w, c = frame.shape

            # 11. DRAW ALL 21 LANDMARKS          

            for lm in hand_landmarks:

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                cv2.circle(
                    frame,
                    (cx, cy),
                    5,
                    (0, 255, 0),
                    -1
                )

    cv2.imshow(
        "Finger Counter",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()