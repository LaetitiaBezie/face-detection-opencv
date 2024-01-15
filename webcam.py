import cv2
import mediapipe as mp

# appliquer le classifieur opencv pour les images avec un visage à face frontal
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Initialise le module pour la reconnaissance des mains
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mp.solutions.drawing_utils

# Capture vidéo à partir de la webcam (0 correspond à la caméra par défaut)
cap = cv2.VideoCapture(0)

# Détecte les visages dans le flux vidéo et dessine une boîte de délimitation autour
def detect_bounding_box_face(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

def detect_bounding_box_hands(vid):
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(vid, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    results = hands.process(rgb_frame)    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame with red color
            mp_drawing.draw_landmarks(
                vid, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            )  

    return results.multi_hand_landmarks  

while True:
    # Capture une image
    result, video_frame = cap.read()  
    if not result:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box_face(video_frame)
    hand_landmarks = detect_bounding_box_hands(video_frame)

    # Affiche l'image capturée
    cv2.imshow("My Face and Hand Detection Project", video_frame)  

    # Sortir de la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
