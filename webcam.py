import cv2

#appliquer le classifieur opencv pour les images avec un visage à face frontal
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# Capture vidéo à partir de la webcam (0 correspond à la caméra par défaut)
cap = cv2.VideoCapture(0)

#Détecte les visages dans le flux vidéo et dessine une boîte de délimitation autour 
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces


while True:
    #Capture une image
    result, video_frame = cap.read()  
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(
        video_frame
    )  # apply the function we created to the video frame

    # Affiche l'image capturée
    cv2.imshow(
        "My Face Detection Project", video_frame
    )  
    # Sortir de la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



