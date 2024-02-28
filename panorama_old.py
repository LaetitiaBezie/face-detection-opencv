import cv2
import numpy as np
import sys

def main(image_paths, output_path, matching_type, fusion_type):
    # Liste pour stocker les images alignées
    aligned_images = []

    # Charge toutes les images et effectue la détection des points d'intérêt et le calcul des descripteurs
    for image_path in image_paths:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        aligned_images.append((image, keypoints, descriptors))

    # Utilisez une image de référence pour aligner les autres images
    reference_image, reference_keypoints, reference_descriptors = aligned_images[0]

    # Boucle sur les images restantes pour les aligner sur l'image de référence
    for image, keypoints, descriptors in aligned_images[1:]:
        if matching_type == 'brute_force':
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(reference_descriptors, descriptors)
        elif matching_type == 'flann':
            index_params = dict(algorithm=0, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.match(reference_descriptors, descriptors)
        else:
            print("Type d'appariement non valide.")
            sys.exit(1)

        num_matches = 50
        matches = sorted(matches, key=lambda x: x.distance)[:num_matches]

        # Extrait les coordonnées des points correspondants
        src_points = np.float32([reference_keypoints[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
        dst_points = np.float32([keypoints[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

        # Estime la matrice d'homographie
        homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

        # Applique l'homographie pour aligner l'image sur l'image de référence
        aligned_image = cv2.warpPerspective(image, homography, (reference_image.shape[1], reference_image.shape[0]))
        aligned_images.append(aligned_image)

    # Fusionne toutes les images alignées en une seule
    blended_image = np.zeros_like(reference_image)
    for aligned_image in aligned_images:
        if fusion_type == 'alpha':
            alpha = 0.5  # blending factor   
            blended_image = cv2.addWeighted(blended_image, alpha, aligned_image, 1-alpha, 0)
        elif fusion_type == 'feathering':
            # la fusion par effet de fondu
            blended_image = cv2.add(blended_image, aligned_image)
        elif fusion_type == 'multi_band':
            #la fusion multi-bandes
                # Divise les images en différentes bandes
            bands1 = cv2.pyrDown(reference_image)
            bands2 = cv2.pyrDown(aligned_image)
            # Fusionne chaque bande
            blended_image = cv2.pyrUp(cv2.addWeighted(bands1, alpha, bands2, 1-alpha, 0))
        elif fusion_type == 'weighted_average':
            # la moyenne pondérée
            blended_image = cv2.addWeighted(blended_image, 0.5, aligned_image, 0.5, 0)

        elif fusion_type == 'crossfade':
            # le fondu enchaîné
                # Crée une série d'images intermédiaires entre les deux images
                num_frames = 10
                for i in range(1, num_frames + 1):
                    alpha = i / (num_frames + 1)
                    blended_frame = cv2.addWeighted(reference_image, alpha, aligned_image, 1 - alpha, 0)
                    cv2.imshow('Blended Frame', blended_frame)
                    cv2.waitKey(100)
                cv2.destroyAllWindows()
                blended_image = blended_frame  # Utilise la dernière image intermédiaire comme image fusionnée
        elif fusion_type == 'gaussian_pyramid':
            #  la pyramide gaussienne
                # Construit les pyramides gaussiennes pour les deux images
                gaussian_pyramid1 = cv2.pyrDown(reference_image)
                gaussian_pyramid2 = cv2.pyrDown(aligned_image)
                # Fusionne les images à partir des pyramides
                blended_image = cv2.addWeighted(gaussian_pyramid1, alpha, gaussian_pyramid2, 1 - alpha, 0)
        else:
            print("Type de fusion non valide.")
            sys.exit(1)

    # Enregistre l'image fusionnée
    cv2.imwrite(output_path, blended_image)
    print(f"Image fusionnée enregistrée sous: {output_path}")

    # Affiche l'image fusionnée
    cv2.imshow('Blended Image', blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 7:
        print("Usage: python fusion_images.py <chemin_image1> <chemin_image2> ... <chemin_imageN> <chemin_output_path>")
        sys.exit(1)
    main(sys.argv[1:-1], sys.argv[-1])
