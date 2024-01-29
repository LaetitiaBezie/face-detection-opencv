import cv2
import numpy as np
import sys
def main(image1_path, image2_path, output_path):

    # Charge les images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Convertir en niveaux de gris
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialise le détecteur de point d'intérêt SIFT
    sift = cv2.SIFT_create()

    # Détecte les points d'intérêts et les descripteurs
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Effectue le matching des descripteurs avec brute force
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sélectionne les 50 premiers appariements
    num_matches = 50
    matches = sorted(matches, key=lambda x: x.distance)[:num_matches]

    # Extrait les coordonnées des points correspondants
    src_points = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

    # Estime la matrice d'homographie
    homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    # Applique l'homographie pour aligner les images
    result = cv2.warpPerspective(image1, homography, (image2.shape[1], image2.shape[0]))

    # Fusionne les deux images alignées
    alpha = 0.5  # blending factor
    blended_image = cv2.addWeighted(result, alpha, image2, 1 - alpha, 0)
    
    # Enregistre l'image fusionnée
    cv2.imwrite(output_path, blended_image)
    print(f"Image fusionnée enregistrée sous: {output_path}")


    # Affiche l'image fusionné
    cv2.imshow('Blended Image', blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python fusion_images.py <chemin_image1> <chemin_image2> <chemin_output_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2],sys.argv[3])