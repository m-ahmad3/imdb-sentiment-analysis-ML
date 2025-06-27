from flask import Flask, render_template, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load models and vectorizer
xgb_model = joblib.load('../xgboost_model.joblib')
lr_model = joblib.load('../logistic_model.joblib')
svc_model = joblib.load('../svc_model.joblib')
tfidf = joblib.load('../tfidf_vectorizer.joblib')

# Text preprocessing functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuations and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stopwords and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        review = data['review']
        model_choice = data['model']
        
        # Preprocess the review
        cleaned_review = clean_text(review)
        
        # Vectorize the review
        review_vectorized = tfidf.transform([cleaned_review])
        
        # Select model and make prediction
        if model_choice == 'xgboost':
            model = xgb_model
        elif model_choice == 'logistic':
            model = lr_model
        else:
            model = svc_model
            
        prediction = model.predict(review_vectorized)[0]
        probability = model.predict_proba(review_vectorized)[0].tolist() if model_choice != 'svc' else None
        
        result = {
            'sentiment': 'Positive' if prediction == 1 else 'Negative',
            'probability': probability[1] if probability else None,
            'success': True
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
