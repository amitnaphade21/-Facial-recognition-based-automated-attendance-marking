import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
import numpy as np
import cv2
from mtcnn import MTCNN
import torch
from facenet_pytorch import InceptionResnetV1

# Initialize models
facenet = InceptionResnetV1(pretrained='vggface2').eval()  # FaceNet
detector = MTCNN()  # MTCNN for face detection

# Load the trained database
database_file = r"A:\SEM 5\EDI\refined_face_embeddings_test.npy"  # Path to saved embeddings
database = np.load(database_file, allow_pickle=True).item()

# Function to compute normalized embeddings for a face
def get_face_embedding(image):
    resized_face = cv2.resize(image, (160, 160))  # Resize to FaceNet input size
    resized_face = resized_face / 255.0  # Normalize pixel values
    resized_face = torch.tensor(resized_face).permute(2, 0, 1).unsqueeze(0).float()  # Convert to tensor
    embedding = facenet(resized_face).detach().numpy()
    embedding = embedding.flatten()  # Ensure the embedding is 1D
    return embedding / np.linalg.norm(embedding)  # Normalize embedding

# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Recognize detected faces and annotate the image
def recognize_faces_with_annotation(group_image_path):
    # Detect faces
    image = cv2.imread(group_image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb_image)
    
    if not detections:
        print("No faces detected!")
        return
    
    # Attendance list
    attendance = []
    recognized_faces = {}
    recognized_names = set()

    for i, detection in enumerate(detections):
        x, y, width, height = detection['box']
        cropped_face = image[y:y+height, x:x+width]
        
        # Debugging: Check cropped face size
        print(f"Face {i+1} cropped size: {cropped_face.shape}")
        
        # Get embedding for the cropped face
        embedding = get_face_embedding(cropped_face)
        
        # Compare with database
        similarity_scores = []
        for name, db_embedding in database.items():
            similarity = cosine_similarity(embedding, db_embedding)
            similarity_scores.append((name, similarity))
        
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        threshold = 0.45
        recognized_name = "Unknown"
        
        if similarity_scores[0][1] >= threshold and similarity_scores[0][0] not in recognized_names:
            recognized_name = similarity_scores[0][0]
            recognized_faces[recognized_name] = similarity_scores[0][1]
            recognized_names.add(recognized_name)
        elif len(similarity_scores) > 1 and similarity_scores[1][1] >= threshold and similarity_scores[1][0] not in recognized_names:
            recognized_name = similarity_scores[1][0]
            recognized_faces[recognized_name] = similarity_scores[1][1]
            recognized_names.add(recognized_name)
        
        attendance.append(recognized_name)
        print(f"Face {i+1} recognized as: {recognized_name} (similarity: {similarity_scores[0][1]:.2f})")
        
        # Draw a rectangle and annotate the face
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 254, 0), 2)
        cv2.putText(image, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 245, 0), 2)
    
    # Save attendance
    attendance_file = os.path.splitext(group_image_path)[0] + "_attendance.txt"
    with open(attendance_file, 'w') as file:
        file.writelines(f"{name}\n" for name in set(attendance) if name != "Unknown")
    print(f"Attendance saved to {attendance_file}")
    
    # Save annotated image
    output_image_path = os.path.splitext(group_image_path)[0] + "_annotated.jpg"
    cv2.imwrite(output_image_path, image)
    print(f"Annotated image saved to {output_image_path}")

# Replace this path with your group photo
group_image_path = r"A:\SEM 5\EDI\Input\Input 5.jpg"  # Path to group photo
recognize_faces_with_annotation(group_image_path)
