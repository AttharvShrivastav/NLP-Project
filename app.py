from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

# Load the model from the models directory
MODEL_PATH = os.path.join('models', 'emotion_classifier_pipe_lr_03_june_2021.pkl')
model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.json.get('text', None)
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Make predictions using the loaded model
        prediction = model.predict([text])[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
