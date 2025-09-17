from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load("time.pkl")

# Route to serve the HTML page
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Route to handle prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        # Extract features
        features = np.array([[data["Sex"], data["Age"], data["Fare"], data["Pclass"]]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Return JSON response
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
