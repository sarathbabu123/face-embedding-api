from flask import Flask, request, jsonify
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import cv2

# Initialize the MTCNN face detection model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Function to generate an embedding from a detected face image
def get_face_embedding(image):
    # Detect faces using MTCNN
    faces, probs = mtcnn.detect(image)

    if faces is None:
        # No face detected
        return None

    embeddings = []
    for face in faces:
        # Crop the face and generate an embedding
        x1, y1, x2, y2 = [int(coord) for coord in face]
        face_crop = image[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, (160, 160))
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_tensor = torch.tensor(face_crop).float().permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(face_tensor).cpu().numpy()
            embeddings.append(embedding)

    return embeddings

# Create Flask app
app = Flask(__name__)

# API endpoint to generate embeddings
@app.route('/generate-embedding', methods=['POST'])
def generate_embedding():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if img is None or img.size == 0:
        return jsonify({'error': 'Invalid image'}), 400

    embeddings = get_face_embedding(img)

    if embeddings is None:
        return jsonify({'error': 'No face detected in the image'}), 400

    # Return the embedding of the first detected face
    return jsonify({'embedding': embeddings[0].tolist()}), 200

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
