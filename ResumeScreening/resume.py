import os
import re
import nltk
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize NLTK stopwords
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def extract_features(text):
    doc = nlp(text)
    features = {
        'num_words': len(doc),
        'num_entities': len(doc.ents),
        'num_verbs': len([token for token in doc if token.pos_ == 'VERB']),
    }
    return features

def load_data(resume_file, job_desc_file):
    with open(resume_file, 'r', encoding='utf-8') as file:
        resume = file.read()
    
    with open(job_desc_file, 'r', encoding='utf-8') as file:
        job_description = file.read()
    
    return [resume], job_description

def create_dataset(resumes, job_description):
    processed_resumes = [preprocess_text(resume) for resume in resumes]
    processed_job_desc = preprocess_text(job_description)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_resumes + [processed_job_desc])
    
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    additional_features = [extract_features(resume) for resume in resumes]
    
    data = pd.DataFrame({
        'resume': resumes,
        'processed_resume': processed_resumes,
        'cosine_similarity': cosine_similarities,
        **{f'feature_{k}': [d[k] for d in additional_features] for k in additional_features[0]}
    })
    
    data['suitable'] = (data['cosine_similarity'] > 0.5).astype(int)
    
    return data, vectorizer

def train_model(data):
    features = ['cosine_similarity', 'feature_num_words', 'feature_num_entities', 'feature_num_verbs']
    X = data[features]
    y = data['suitable']
    
    if len(X) > 1:  # Only split if there's more than one sample
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        # If there's only one sample, use it for both training and testing
        X_train, y_train = X, y
        X_test, y_test = X, y  # Use the same for testing

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model

def screen_resume(resume_text, job_description, model, vectorizer):
    processed_resume = preprocess_text(resume_text)
    processed_job_desc = preprocess_text(job_description)
    
    # Transform the text into tf-idf vectors
    tfidf_matrix = vectorizer.transform([processed_resume, processed_job_desc])
    cosine_similarity_score = cosine_similarity(tfidf_matrix[1], tfidf_matrix[0]).flatten()[0]
    
    # Extract additional features from the resume text
    additional_features = extract_features(resume_text)
    
    # Create a DataFrame with the extracted features
    features = pd.DataFrame({
        'cosine_similarity': [cosine_similarity_score],
        'feature_num_words': [additional_features['num_words']],
        'feature_num_entities': [additional_features['num_entities']],
        'feature_num_verbs': [additional_features['num_verbs']]
    })
    
    # Make a prediction using the trained model
    prediction = model.predict(features)[0]
    
    # Check if the model can predict probabilities and handle the single-class scenario
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        # If there is only one class, set probability to the first element or handle gracefully
        probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
    else:
        probability = None  # Set to None if probabilities cannot be calculated
    
    return prediction, probability

# Main execution
if __name__ == "__main__":
    resume_file = '/content/resume.txt'  # Path to your resume text file
    job_desc_file = '/content/job_description.txt'  # Path to your job description text file
    
    resumes, job_description = load_data(resume_file, job_desc_file)
    data, vectorizer = create_dataset(resumes, job_description)
    model = train_model(data)
    
    # Save the model and vectorizer for future use
    joblib.dump(model, 'resume_screening_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    
    # Example of screening a new resume
    new_resume = """
    Jane Smith  
456 Maple Ave, Austin, TX 78702  
(512) 987-6543  
janesmith@example.com  

Objective:  
To obtain a challenging software engineering position that leverages my expertise in full-stack development and problem-solving skills.

Education:  
Bachelor of Science in Computer Science  
University of Texas at Austin, May 2024  
- Relevant Coursework: Data Structures, Algorithms, Software Engineering, Machine Learning, Web Development

Experience:  
Software Developer Intern  
Innovative Tech Solutions, Austin, TX  
May 2023 - August 2023  
- Developed and maintained web applications using React, Node.js, and Express.  
- Collaborated with a team of 5 developers to design and implement a microservices architecture for a high-traffic e-commerce site.  
- Implemented RESTful APIs that improved data processing speed by 30%.  
- Conducted code reviews and implemented unit testing using Jest and Mocha.

Technical Support Intern  
Tech Co, Austin, TX  
June 2022 - August 2022  
- Provided technical support and troubleshooting for web-based applications.  
- Developed scripts in Python to automate routine data processing tasks, reducing manual workload by 50%.

Projects:  
Personal Portfolio Website  
- Created a personal website to showcase projects using HTML, CSS, and JavaScript.  
- Integrated a blog section using a headless CMS and optimized the website for SEO.

E-commerce Web Application  
- Developed a full-stack application using React, Node.js, MongoDB, and Express.  
- Implemented payment integration and user authentication with JWT.

Skills:  
- Programming Languages: Python, Java, JavaScript, HTML, CSS  
- Frameworks: React, Node.js, Express, Flask  
- Tools: Git, Docker, AWS, Jenkins  
- Databases: MySQL, MongoDB  

Certifications:  
- AWS Certified Solutions Architect  
- FreeCodeCamp Full Stack Web Development Certification

Awards:  
- Dean’s List, 2022-2024  
- Winner, HackUT, Best Web Application, 2023

Extracurricular Activities:  
- Vice President, University Coding Club  
- Mentor, Women in Tech Initiative  

References:  
Available upon request.
""" # Replace with actual new resume text
    prediction, probability = screen_resume(new_resume, job_description, model, vectorizer)
    
    print(f"\nNew Resume Screening Result:")
    print(f"Suitable: {'Yes' if prediction == 1 else 'No'}")
    print(f"Confidence: {probability:.2f}" if probability is not None else "Confidence: N/A")
