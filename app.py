from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Global variables for model and encoders
model = None
encoders = None
COUNTRIES = []
TOPICS = []
SUBTOPICS = []
LEVELS = []

def load_model_and_data():
    """Load model, encoders, and dataset categories once at startup"""
    global model, encoders, COUNTRIES, TOPICS, SUBTOPICS, LEVELS
    
    try:
        # Load the trained model and encoders
        print("Loading model and encoders...")
        with open('models/mathe_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        print("✅ Model and encoders loaded successfully!")
        
        # Load dataset to get exact categories used in training
        print("Loading dataset categories...")
        df_categories = pd.read_csv('MathE dataset.csv', delimiter=';', encoding='latin-1')
        
        # Get exact categories from the dataset (same as used in training)
        COUNTRIES = sorted(df_categories['Student Country'].unique().tolist())
        TOPICS = sorted(df_categories['Topic'].unique().tolist())
        SUBTOPICS = sorted(df_categories['Subtopic'].unique().tolist())
        LEVELS = sorted(df_categories['Question Level'].unique().tolist())
        
        print(f"✅ Categories loaded: {len(COUNTRIES)} countries, {len(TOPICS)} topics")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model or data: {e}")
        return False

# Load everything at startup
print("🚀 Starting Flask app...")
if not load_model_and_data():
    print("⚠️  Warning: Model or data not loaded properly")

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    if model is not None and encoders is not None:
        return {"status": "healthy", "model_loaded": True}, 200
    else:
        return {"status": "unhealthy", "model_loaded": False}, 503

@app.route('/')
def index():
    """Home page with prediction form"""
    if model is None or encoders is None:
        return render_template('result.html', 
                             error="Application is starting up. Please wait a moment and refresh.")
    
    return render_template('index.html', 
                         countries=COUNTRIES,
                         topics=TOPICS, 
                         subtopics=SUBTOPICS,
                         levels=LEVELS)

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on form input"""
    try:
        # Get form data
        country = request.form['country']
        level = request.form['level']
        topic = request.form['topic']
        subtopic = request.form['subtopic']
        
        # Check if model is loaded
        if model is None or encoders is None:
            return render_template('result.html', 
                                 error="Model not loaded properly. Please try again later.")
        
        # Create feature array
        # Check if all categories are known to the encoders
        try:
            country_encoded = encoders['country'].transform([country])[0]
        except ValueError:
            return render_template('result.html', 
                                 error=f"Unknown country: {country}. Available countries: {', '.join(COUNTRIES)}")
        
        try:
            level_encoded = encoders['level'].transform([level])[0]
        except ValueError:
            return render_template('result.html', 
                                 error=f"Unknown level: {level}. Available levels: {', '.join(LEVELS)}")
        
        try:
            topic_encoded = encoders['topic'].transform([topic])[0]
        except ValueError:
            return render_template('result.html', 
                                 error=f"Unknown topic: {topic}. Available topics: {', '.join(TOPICS)}")
        
        try:
            subtopic_encoded = encoders['subtopic'].transform([subtopic])[0]
        except ValueError:
            return render_template('result.html', 
                                 error=f"Unknown subtopic: {subtopic}. Available subtopics: {', '.join(SUBTOPICS)}")
        
        # Create feature vector
        features = np.array([[country_encoded, level_encoded, topic_encoded, subtopic_encoded]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Get confidence score
        confidence = max(probability) * 100
        
        # Prepare result
        result = "Correct" if prediction == 1 else "Incorrect"
        
        return render_template('result.html',
                             country=country,
                             level=level,
                             topic=topic,
                             subtopic=subtopic,
                             prediction=result,
                             confidence=round(confidence, 1))
                             
    except Exception as e:
        return render_template('result.html', 
                             error=f"Error making prediction: {str(e)}")

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Flask app on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
