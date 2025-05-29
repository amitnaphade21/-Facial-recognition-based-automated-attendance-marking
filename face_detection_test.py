import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
from mtcnn import MTCNN
import torch
from sklearn.ensemble import IsolationForest

# Initialize models
facenet = InceptionResnetV1(pretrained='vggface2').eval()  # FaceNet model
detector = MTCNN()  # MTCNN for face detection

# Directory paths
dataset_dir = r"A:\SEM 5\EDI\data"
embeddings = {}

def preprocess_image(image):
    """
    Detect the largest face in the image, preprocess it (normalize, align), and return it.
    """
    # Convert to RGB for MTCNN
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    detections = detector.detect_faces(rgb_image)
    if not detections:
        return None  # No face detected

    # Find the largest face based on bounding box area
    largest_face = max(detections, key=lambda d: d['box'][2] * d['box'][3])
    x, y, width, height = largest_face['box']

    # Crop the face region
    cropped_face = image[y:y + height, x:x + width]

    # Normalize illumination using histogram equalization
    grayscale = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(grayscale)
    cropped_face[:, :, 0] = equalized

    # Resize the face to 160x160 (FaceNet input size)
    resized_face = cv2.resize(cropped_face, (160, 160))
    resized_face = resized_face / 255.0  # Normalize pixel values
    return resized_face

def augment_image(image):
    """
    Generate augmented versions of the input image (rotations, flipping, zooming).
    """
    augmented_images = []
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Rotations
    angles = [-15, -10, 0, 10, 15]
    for angle in angles:
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        augmented_images.append(rotated)

    # Flipping
    augmented_images.append(cv2.flip(image, 1))  # Horizontal flip

    # Zooming
    scale_factors = [1.1, 1.2]
    for scale in scale_factors:
        zoomed = cv2.resize(image, None, fx=scale, fy=scale)
        zoomed = zoomed[(zoomed.shape[0] - h) // 2:(zoomed.shape[0] + h) // 2,
                        (zoomed.shape[1] - w) // 2:(zoomed.shape[1] + w) // 2]
        augmented_images.append(zoomed)

    return augmented_images

def generate_embeddings(image):
    """
    Generate embeddings for the input image and its augmentations.
    """
    augmented_images = augment_image(image)
    embeddings = []

    for aug_image in augmented_images:
        aug_image_tensor = torch.tensor(aug_image).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            embedding = facenet(aug_image_tensor).detach().numpy().flatten()
        embeddings.append(embedding / np.linalg.norm(embedding))  # Normalize embeddings

    return embeddings

# Process each person's folder
for person_name in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_name)

    if os.path.isdir(person_path):
        print(f"Processing {person_name}")
        person_embeddings = []

        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)
            image = cv2.imread(image_path)

            if image is None:
                continue

            preprocessed = preprocess_image(image)
            if preprocessed is not None:
                embeddings_for_image = generate_embeddings(preprocessed)
                person_embeddings.extend(embeddings_for_image)

        if person_embeddings:
            # Remove outliers using Isolation Forest
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = isolation_forest.fit_predict(person_embeddings)
            filtered_embeddings = [
                embedding for i, embedding in enumerate(person_embeddings) if predictions[i] == 1
            ]

            # Store the mean of filtered embeddings for the person
            if filtered_embeddings:
                embeddings[person_name] = np.mean(filtered_embeddings, axis=0)

# Save refined embeddings
np.save(r"A:\SEM 5\EDI\refined_face_embeddings_test.npy", embeddings)
print("Database embeddings refined and saved.")
