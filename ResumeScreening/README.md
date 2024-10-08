# Resume Screening Project

This project aims to automate the screening of resumes based on job descriptions using Natural Language Processing (NLP) and Machine Learning techniques. The system preprocesses resumes and job descriptions, extracts relevant features, calculates similarities, and predicts the suitability of a candidate for a given job.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Example](#example)
- [Model Training](#model-training)
- [Saving and Loading Models](#saving-and-loading-models)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project uses:
- Natural Language Processing (NLP) libraries such as NLTK and spaCy for text preprocessing and feature extraction.
- TF-IDF Vectorizer to calculate the textual similarity between resumes and job descriptions.
- A RandomForestClassifier model to classify resumes as suitable or not based on extracted features.
- Cosine similarity to measure how closely the resume matches the job description.

The goal is to assist in screening resumes quickly and efficiently, helping recruiters focus on the most promising candidates.

## Features

- **Preprocessing**: Cleans and tokenizes text data from resumes and job descriptions.
- **Feature Extraction**: Extracts important features like word count, number of entities, and verbs.
- **Cosine Similarity**: Measures the similarity between the resume and the job description.
- **Classification**: Trains a machine learning model to classify resumes as suitable or not.
- **Model Persistence**: Saves trained models and vectorizers for future use.

## Requirements

- Python 3.8 or higher
- Required libraries:
  - `nltk`
  - `spacy`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `joblib`

You can install the required libraries using the command:

```bash
pip install nltk spacy scikit-learn pandas numpy joblib
Download the necessary NLTK data:
python
Copy code
import nltk
nltk.download('punkt')
nltk.download('stopwords')
Download and install the spaCy model:
bash
Copy code
python -m spacy download en_core_web_sm
Setup
Clone the repository or download the project files.

Ensure you have the required libraries and data installed as mentioned above.

Place your resume and job description text files in the project directory.

Usage
Running the Script
Prepare your input files:

Save your resume as resume.txt.
Save the job description as job_description.txt.
Run the main script:

bash
Copy code
python main.py
Screen a New Resume: You can use the screen_resume() function to test a new resume against the saved job description using the trained model.

Input and Output
Input:

Text files containing the resume and job description.
Output:

Prints a classification report and confusion matrix during training.
Displays the screening result (suitable or not) for a new resume with confidence score if available.
Example
Example New Resume Screening
Here’s how you can screen a new resume:

python
Copy code
# Replace with the path to your new resume text
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

Skills:
- Programming Languages: Python, Java, JavaScript, HTML, CSS
- Frameworks: React, Node.js, Express, Flask
- Tools: Git, Docker, AWS, Jenkins
- Databases: MySQL, MongoDB
"""
prediction, probability = screen_resume(new_resume, job_description, model, vectorizer)

print(f"\nNew Resume Screening Result:")
print(f"Suitable: {'Yes' if prediction == 1 else 'No'}")
print(f"Confidence: {probability:.2f}" if probability is not None else "Confidence: N/A")
Model Training
The train_model() function handles model training, including splitting the data, training a RandomForestClassifier, and printing evaluation metrics.

Saving and Loading Models
Models and vectorizers are saved using joblib to allow for reuse without retraining.
Load the model and vectorizer using joblib.load('model_filename').
Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

