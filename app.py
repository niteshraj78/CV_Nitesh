from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/upload'
OUTPUT_FOLDER = 'static/output'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def apply_transformations(image_path):
    """Applies scaling, translation, rotation, and shearing to the image."""
    image = cv2.imread(image_path)
    rows, cols, _ = image.shape

    transformations = {}

    # Scaling
    scaling_matrix = np.array([[1.5, 0, 0],
                                [0, 1.5, 0]], dtype=np.float32)
    transformations['Scaling'] = cv2.warpAffine(image, scaling_matrix, (cols, rows))

    # Translation
    translation_matrix = np.array([[1, 0, 50],
                                    [0, 1, 100]], dtype=np.float32)
    transformations['Translation'] = cv2.warpAffine(image, translation_matrix, (cols, rows))

    # Rotation
    center = (cols // 2, rows // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1)  # 45 degrees
    transformations['Rotation'] = cv2.warpAffine(image, rotation_matrix, (cols, rows))

    # Shearing
    shearing_matrix = np.array([[1, 0.5, 0],
                                 [0.5, 1, 0]], dtype=np.float32)
    transformations['Shearing'] = cv2.warpAffine(image, shearing_matrix, (cols, rows))

    # Save transformed images
    output_paths = {}
    for name, transformed_image in transformations.items():
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{name}.jpg")
        cv2.imwrite(output_path, transformed_image)
        output_paths[name] = f"/{output_path}"  # Relative path for HTML display

    return output_paths


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Apply transformations
            output_paths = apply_transformations(file_path)

            # Render results
            return render_template('result.html', images=output_paths)
    return render_template('index.html')


@app.route('/home', methods=['GET'])
def home():
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
