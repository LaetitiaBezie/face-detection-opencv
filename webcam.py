import cv2
import mediapipe as mp

#classifieur opencv pour les images avec un visage à face frontal
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# initialise le module pour la reconnaissance des mains
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# capture vidéo à partir de webcam
cap = cv2.VideoCapture(0)

# détecte les visages dans le flux vidéo et dessine une boîte de délimitation autour
def detect_bounding_box_face(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

# détecte les mains dans le flux vidéo et dessine le squelette correspondant
def detect_bounding_box_hands(vid):
    rgb_frame = cv2.cvtColor(vid, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                vid, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            )  

    return results.multi_hand_landmarks  

while True:
    # capture une image
    result, video_frame = cap.read()  
    if not result:
        break  

    faces = detect_bounding_box_face(video_frame)
    hand_landmarks = detect_bounding_box_hands(video_frame)

    # affiche l'image capturée
    cv2.imshow("My Face and Hand Detection Project", video_frame)  

    # sortir de la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
