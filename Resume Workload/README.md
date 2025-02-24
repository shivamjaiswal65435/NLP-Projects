# 📄 Resume Classification using Machine Learning

## 📌 Overview
This project focuses on **automated resume classification** using **Natural Language Processing (NLP) and Machine Learning**. The model categorizes resumes based on job roles (e.g., Data Scientist, Software Engineer, Marketing, etc.), helping recruiters quickly filter candidates.

## 🚀 Features
- **Resume Parsing**: Extracts structured information from resumes
- **Text Preprocessing**: Tokenization, stopword removal, lemmatization
- **Feature Engineering**: TF-IDF vectorization, word embeddings
- **ML Models**: Logistic Regression, Naïve Bayes, Random Forest, Deep Learning
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score
- **Web App Integration**: Upload a resume and get instant classification

## 📂 Dataset
- The dataset consists of resumes labeled by job categories
- Resumes are preprocessed to extract relevant keywords and skills

## 🛠️ Technologies Used
- **Python** 🐍
- **Libraries**: Pandas, Numpy, Scikit-Learn, TensorFlow, NLTK, Spacy
- **NLP & Machine Learning**

## 📊 Model Performance
| Model               | Accuracy  |
|--------------------|----------|
| Logistic Regression | 88%      |
| Naïve Bayes        | 85%      |
| Random Forest      | 90%      |

## 🔧 Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/shivamjaiswal65435/resume-classification.git
   cd resume-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script to train the model:
   ```bash
   python train.py
   ```
4. Classify a resume:
   ```bash
   python predict.py --resume "path/to/resume.pdf"
   ```

## 📌 Usage
- Automate resume screening for recruiters
- Enhance job-matching efficiency in HR systems
- Deploy as a web app for real-time classification

## 🔗 Connect with Me
👨‍💻 **Shivam Jaiswal**  
🔗 [GitHub](https://github.com/shivamjaiswal65435)  
🔗 [LinkedIn](https://www.linkedin.com/in/shivam-jaiswal65425/)  

📢 Feel free to fork, contribute, or raise issues! 🚀
