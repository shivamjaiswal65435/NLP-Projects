import streamlit as st
import base64
import pickle
import PyPDF2  # for PDF text extraction
import docx2txt  # for DOCX text extraction
from sklearn.feature_extraction.text import TfidfVectorizer  # for TF-IDF vectorization

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


add_bg_from_local('certification_on_resume.png')

# Load the SVM model from the pickle file
with open('SVMP.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Load the TF-IDF vectorizer used during training
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Define a dictionary to map numerical labels to meaningful categories
label_to_category = {
    0: "PeopleSoft Resumes",
    1: "ReactJS Developer",
    2: "SQL Developer Lightning Insight",
    3: "Workday Resume",
    # Add more mappings as needed
}

# Streamlit app title
st.markdown("<h1 style='color: BLACK;'>RESUME CLASSIFICATION APP</h1>", unsafe_allow_html=True)


# File uploader for resumes (PDF and Word)
uploaded_files = st.file_uploader("Upload one or more resumes (PDF or Word):", type=["pdf", "docx"], accept_multiple_files=True)

# Classification button
if st.button('Classify') and uploaded_files:
    for uploaded_file in uploaded_files:
        # Extract text from the uploaded file (PDF or DOCX)
        file_extension = uploaded_file.name.split('.')[-1].lower()
        resume_text = ""
        if file_extension == "pdf":
            pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
            for page_num in range(pdf_reader.getNumPages()):
                resume_text += pdf_reader.getPage(page_num).extractText()
        elif file_extension == "docx":
            resume_text = docx2txt.process(uploaded_file)
        
        if resume_text:
            # Preprocess the resume text using the loaded TF-IDF vectorizer
            preprocessed_resume = tfidf_vectorizer.transform([resume_text])
            
            # Make predictions using the SVM model
            numerical_label = svm_model.predict(preprocessed_resume)[0]
            
            # Map the numerical label to a meaningful category
            category = label_to_category.get(numerical_label, "Unknown")
            
            # Display the predicted category for the current uploaded resume
            st.write(f'Predicted Category for {uploaded_file.name}: {category}')
        else:
            st.write(f'Error: Failed to extract text from {uploaded_file.name}')

