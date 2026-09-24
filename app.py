from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the scikit-learn SVC (Support Vector Classifier) model.
# Note: Ensure your serialized model file is named 'model.pkl' and placed in the root directory.
# The source model utilizes a 'linear' kernel and is formatted via joblib[cite: 1].
MODEL_PATH = 'anemia_svm_model.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', error_text='Error: Model file (model.pkl) not found.')
    
    try:
        # Extract features from the submitted form
        # Adjust the number of inputs if your SVC model requires a different feature count
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        # Generate prediction
        prediction = model.predict(final_features)
        
        # The model returns an array of classes[cite: 1]
        predicted_class = prediction[0]
        
        return render_template('index.html', prediction_text=f'Predicted Class: {predicted_class}')
    except ValueError:
        return render_template('index.html', error_text='Error: Please enter valid numerical values.')
    except Exception as e:
        return render_template('index.html', error_text=f'An error occurred: {str(e)}')

if __name__ == "__main__":
    # Render assigns the PORT environment variable dynamically
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
