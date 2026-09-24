# Anemia Detection Model

Live Demo: https://anaemia-detection-image-classification.onrender.com

A minimal web application built with Flask that predicts anemia from medical images using a pre-trained Support Vector Machine (SVM) classifier. The application processes uploaded images, extracts 3D color histograms using OpenCV, and serves predictions via a clean, responsive Tailwind CSS interface.

## Features

* **Image Processing:** Utilizes OpenCV for resizing, color conversion, and extracting 512-dimensional color histogram features.
* **Data Augmentation:** Implements Albumentations for robust preprocessing.
* **Minimal UI:** Features a responsive, drag-and-drop file upload interface built with Tailwind CSS.
* **Deployment Ready:** Configured for seamless deployment on cloud platforms like Render using Gunicorn.

## Prerequisites

Ensure you have Python 3.8+ installed. The project relies on several key libraries, notably `opencv-python-headless` which is required for headless Linux environments (like Render) to avoid GUI dependency errors.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/anemia-detection-model.git
cd anemia-detection-model

```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

```

3. Install the required dependencies:

```bash
pip install -r requirements.txt

```

4. Ensure your trained model file `anemia_svm_model.pkl` is located in the root directory of the project.

## Local Development

To run the application locally, execute the following command:

```bash
python app.py

```

The application will be accessible at `http://localhost:5000`.

## Deployment to Render

This project is structured to be deployed on Render as a Web Service.

1. Create a new Web Service on Render and connect your GitHub repository.
2. Set the **Build Command** to:

```bash
pip install -r requirements.txt

```

3. Set the **Start Command** to:

```bash
gunicorn app:app

```

4. Render will automatically assign a `PORT` environment variable, which the application is already configured to use.

## Project Structure

```text
├── app.py                      # Main Flask application and inference logic
├── anemia_svm_model.pkl        # Serialized scikit-learn SVM model
├── requirements.txt            # Python dependencies
├── static/
│   └── uploads/                # Temporary storage for processed images
└── templates/
    └── index.html              # Frontend user interface

```
