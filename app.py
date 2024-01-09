from flask import Flask, request, jsonify, render_template
from flask_restful import Api, Resource
import numpy as np
from PIL import Image
import io
import cv2
from skimage.feature import hog
from joblib import load
import pandas as pd

app = Flask(__name__)
api = Api(app)

# Load the trained KNN model and other data
model_path = r'C:\Users\PC\Desktop\Product recommendation Imagae classificaion\knn_model_4.pkl'
knn_model = load(model_path)

csv_file_path = r"C:\Users\PC\Desktop\Product recommendation Imagae classificaion\Data ID - Sheet1.csv"
product_images = pd.read_csv(csv_file_path)

# Function to extract features for an image
def extract_features(image):
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image

    try:
        features, hog_image = hog(gray_img, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
        if len(features) == 0:
            features = np.zeros(1764)  # Placeholder feature vector with 1764 dimensions
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        features = np.zeros(1764)  # Placeholder feature vector with 1764 dimensions

    return features

# Function to preprocess and extract features from the user-uploaded image
def preprocess_user_image(user_image):
    try:
        img = Image.open(io.BytesIO(user_image.read()))
        img = img.resize((224, 224))  # Adjust the image size as needed
        img_array = np.array(img)
        user_input_features = extract_features(img_array)
        return user_input_features
    except Exception as e:
        print(f"Error processing user image: {e}")
        return None

# Function to recommend products based on user input image
def recommend_products(user_image):
    user_input_features = preprocess_user_image(user_image)
    if user_input_features is not None:
        user_input_features = [user_input_features]
        distances, indices = knn_model.kneighbors(user_input_features)
        return indices[0]
    else:
        return []

class RecommendationAPI(Resource):
    def post(self):
        try:
            if 'image_file' not in request.files:
                return {"error": "No file part"}

            file = request.files['image_file']
            if file.filename == '':
                return {"error": "No selected file"}

            recommended_ids = recommend_products(file)
            return {"recommended_ids": recommended_ids}

        except Exception as e:
            return {"error": f"An error occurred: {e}"}

# Define the API endpoint
api.add_resource(RecommendationAPI, '/api/recommend')

@app.route('/', methods=['GET', 'POST'])
def index():
    show_upload = True
    recommended_ids = []

    if request.method == 'POST':
        if 'image_file' not in request.files:
            return "No file part"

        file = request.files['image_file']
        if file.filename == '':
            return "No selected file"

        recommended_ids = recommend_products(file)
        if len(recommended_ids) > 0:
            show_upload = True  # Keep the upload button visible
        else:
            show_upload = False  # Hide the upload button if no recommendations are found

    return render_template('perfume_recommendation.html', show_upload=show_upload, recommended_ids=recommended_ids, product_images=product_images)

if __name__ == '__main__':
    app.run(debug=True)
