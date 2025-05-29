# 🧠 Facial Recognition-Based Automated Attendance System

An automated attendance marking system using facial recognition. This project uses **MTCNN** for face detection and **FaceNet (InceptionResnetV1)** for face recognition to detect and identify students from a group photo and mark their attendance.

---

## 📸 Demo Output

> - 📷 Input group image
>
> - ![Input 1](https://github.com/user-attachments/assets/c9c8df88-816c-4b2c-b115-4bd9ea35b537)

> - 🟩 Output with bounding boxes and names
>
> - ![Input 1_annotated](https://github.com/user-attachments/assets/c3e3d088-da22-40ed-9368-0ed8e51dbc4c)

>
> - ![Screenshot 2025-05-29 180743](https://github.com/user-attachments/assets/60cb0d12-9bf9-4e6c-b5d6-85d770036773)


<p align="center">
  <img src="screenshots/input.jpg" width="45%">
  <img src="screenshots/output.jpg" width="45%">
</p>

---

## 🚀 Features

- Real-time or static group image recognition
- Face embedding and matching using cosine similarity
- Automatic attendance marking and saving to `.txt`
- Image annotation with recognized names

---

## 📂 Project Structure

```bash
Facial-Recognition-Attendance/
│
├── face_detection_and_annotation.py    # Main code for detection + annotation
├── face_recognition_only.py            # Recognition-only version
├── refined_face_embeddings_test.npy    # Precomputed database of known face embeddings
├── Input/                              # Group photos for recognition
├── Output/                             # Annotated images and attendance text
├── screenshots/                        # [Recommended] Save result screenshots here
└── README.md



🧠 Working Principle
📸 Load input group image

🔍 Detect faces using MTCNN

📐 Extract and preprocess each face

🎯 Generate embeddings with FaceNet

📊 Compare embeddings using cosine similarity

🟢 Annotate recognized faces and mark attendance

📁 Save annotated image and text file


##Architecture Diagram:

> - ![Architecture](https://github.com/user-attachments/assets/848bdb0f-5417-40e7-b4bb-67e14958f06a)


📊 Results & Accuracy
Image Sample	Total Faces	Recognized	Accuracy (%)
Input 1.jpg	        5	      5	          100%
Input 2.jpg	        8	      7	          87.5%
Input 3.jpg       	6      	6	          100%



⚙️ How to Run
1. Clone the Repository
git clone https://github.com/amitnaphade21/-Facial-recognition-based-automated-attendance-marking.git
cd -Facial-recognition-based-automated-attendance-marking


2. Install Requirements

pip install -r requirements.txt

pip install opencv-python facenet-pytorch numpy mtcnn torch


3. Run Recognition

# Face recognition only
python face_recognition_only.py

# Recognition + annotation
python face_detection_and_annotation.py

🧾 Input Dataset
Precomputed database: refined_face_embeddings_test.npy


🛠️ Future Enhancements
🎥 Support for live webcam attendance

🌐 Deploy as a web or mobile app

📈 Improve threshold adaptively for large datasets

🧪 Add a GUI using Tkinter or PyQt

🙏 Credits
Facenet-PyTorch

MTCNN paper & implementation

Original VGGFace2 pretraining weights



