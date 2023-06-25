from flask import Flask, request, jsonify
import pickle
import numpy as np
import joblib
app = Flask(__name__)

# Load the trained SVC model

model = joblib.load('/Users/uchhasdewan/practice projects/flask/svm_model_cov_update.pkl')

print(type(model))

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Extract the values from the JSON data
    values = list(data.values())

    # Convert the values to a 2D array
    features = np.array(values).reshape(1, -1)

    # Make predictions using the SVC model
    prediction = model.predict(features)

    # Convert the prediction to a list
    prediction = prediction.tolist()

    # Prepare the response JSON
    response = {
        'prediction': prediction
    }

    # Return the response as JSON
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
