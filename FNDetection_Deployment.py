import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go
import base64
import pyttsx3
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Function for welcome message with increased speech speed
def welcome_message(speak=True):
    if speak:
        welcome_text = "Welcome to the News Detection App. Please enter the article text below."
        st.write(welcome_text)

        # Increase the speed of the speech by setting a higher rate value
        engine = pyttsx3.init()
        engine.setProperty("rate", 180)  # Adjust the rate value as needed
        engine.say(welcome_text)
        engine.runAndWait()

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


add_bg_from_local('fact.jpg')

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)


# Load the trained models and TF-IDF vectorizers
nb_classifier = joblib.load('nb_model.pkl')
lr_classifier = joblib.load('lr_model.pkl')
rf_classifier = joblib.load('rf_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to handle voice output
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 120)  
    engine.say(text)
    engine.runAndWait()

# Streamlit app
def main():
    st.title("Article Prediction App")
   
    if "welcome_message_spoken" not in st.session_state:
        st.session_state.welcome_message_spoken = False

    # Welcome message (only spoken when the app starts)
    welcome_message(speak=not st.session_state.welcome_message_spoken)
    st.session_state.welcome_message_spoken = True

    st.write("<p style='font-size: 24px; font-family: Arial, sans-serif; color: #333;'>This app predicts whether an article is true or false based on its content.</p>", unsafe_allow_html=True)

    # Create a sidebar for model selection
    st.sidebar.write("## Model Selection")
    selected_model = st.sidebar.selectbox("Select Model:", ("Naive Bayes", "Logistic Regression", "Random Forest"))

    # Initialize user_prediction and prediction_probability outside of the if-block
    user_prediction = None
    prediction_probability = None

    # Text input area for user to input article text
    user_input = st.sidebar.text_area("Article Text", "")
    predict_button = st.sidebar.button("Predict")
    

    if predict_button:
        if user_input:
            if selected_model == "Naive Bayes":
                model = nb_classifier
               # st.sidebar.image("Naive Bayes.png", caption="Naive Bayes", use_column_width=False, width=500)
            elif selected_model == "Logistic Regression":
                model = lr_classifier
                # st.sidebar.image("Logistic Regression.png", caption="Logistic Regression", use_column_width=False, width=500)
            elif selected_model == "Random Forest":
                model = rf_classifier
                # st.sidebar.image("Random Forest.png", caption="Random Forest", use_column_width=False, width=500)
            user_input_tfidf = tfidf_vectorizer.transform([user_input])
            user_prediction = model.predict(user_input_tfidf)
            prediction_probability = model.predict_proba(user_input_tfidf)[0][1]
            st.sidebar.write("Predicted Outcome:", "Real news" if user_prediction[0] == 1 else "Fake news")
            st.sidebar.write("Prediction Probability:", f"{prediction_probability:.2f}")
             # Convert prediction information to speech
            prediction_text = "Real news" if user_prediction[0] == 1 else "Fake news"
            probability_text = f"{prediction_probability:.2f}"
            speak_text(f"The predicted outcome is {prediction_text} with a probability of {probability_text}")

            st.sidebar.image(selected_model+".png", caption=selected_model, use_column_width=False, width=500)
            st.write("### Word Cloud for Input Text")
            generate_wordcloud(user_input)
            

                      
    else:
        # Create dynamic graph using Plotly when prediction is not made
        st.write("### Model Performance Comparison")
        models = ["Naive Bayes", "Logistic Regression", "Random Forest"]
        accuracy_scores = [93.29, 98.75, 99.73]  # Example accuracy scores, replace with actual scores

        # Create a bar chart using Plotly
        fig = go.Figure(data=[go.Bar(x=models, y=accuracy_scores)])
        fig.update_layout(xaxis_title="Models", yaxis_title="Accuracy", title="Model Performance Comparison")

        # Customize the colors to black and white
        fig.update_traces(marker_color="white", marker_line_color="black", marker_line_width=1, opacity=0.8)

        # Set the background color of the plot to black
        fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")

        # Show the interactive Plotly chart using st.plotly_chart
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
