import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as ai

# Load the symptom classification model and tokenizer
model_name = st.secrets["repo"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
symptom_model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Symptom mapping dictionary
symptom_mapping = {
    0: "Back pain", 1: "Body feels weak", 2: "Hair falling out", 3: "Heart hurts", 
    4: "Joint pain", 5: "Knee pain", 6: "Head ache", 7: "Infected wound", 8: "Ear ache", 
    9: "Injury from sports", 10: "Feeling cold", 11: "Skin issue", 12: "Neck pain", 
    13: "Cough", 14: "Shoulder pain", 15: "Emotional pain", 16: "Feeling dizzy", 
    17: "Foot ache", 18: "Internal pain", 19: "Out of context", 20: "Acne", 
    21: "Stomach ache", 22: "Blurry vision", 23: "Open wound", 24: "Hard to breathe", 
    25: "Muscle pain"
}

# Function to make predictions and get top symptoms
def predict_top_symptoms(input_text, top_k=3):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = symptom_model(**inputs)
    
    logits = outputs.logits
    # Get the top k predictions
    top_values, top_indices = torch.topk(logits, top_k)
    
    symptoms = []
    for i in range(top_k):
        symptom_index = top_indices[0][i].item()
        symptom_value = top_values[0][i].item()
        predicted_symptom = symptom_mapping.get(symptom_index, "Unknown symptom")
        symptoms.append((predicted_symptom, symptom_value))
    
    return symptoms

# API Key for Google Generative AI
API_KEY = st.secrets["api_key"]

# Configure the Google Generative AI
ai.configure(api_key=API_KEY)

# Streamlit UI setup
st.title("💬 Medimate - Your Medical Chatbot")

# Add a button to restart the conversation
if st.button("Restart Chat"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I assist you with your medical concerns?"}]

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I assist you with your medical concerns?"}]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input handling
if prompt := st.chat_input("Ask a medical question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Predict the top symptoms
    top_symptoms = predict_top_symptoms(prompt)
    symptom_str = ", ".join([f"{symptom} (score: {score:.4f})" for symptom, score in top_symptoms])
    
    # Format the chatbot input
    chatbot_input = f"""
You are Medimate, a medical chatbot. The user has asked the following question:
"{prompt}"

Here are the top predicted symptoms based on the user's input: {symptom_str}

1. If the message is related to medical issues, respond with the following:
    - A list of possible symptoms based on the input.
    - Potential diseases or conditions that could be related to these symptoms.
    - Recommended treatments or next steps.
    - The type of medical specialist the user should consult if necessary.

2. If the message is not related to a medical issue, kindly respond with: 
    "I am a medical chatbot, and I can assist you with health-related concerns. Please ask me a medical question."

Make sure the response is concise, clear, and directly addresses the user's input.
"""

    # Create a new model for the generative response
    chat_model = ai.GenerativeModel("gemini-pro")  # or bison
    chat = chat_model.start_chat()  # Initialize chat
    
    # Send the message along with the top symptoms to the chatbot
    response = chat.send_message(chatbot_input)
    
    # Extract the response content
    msg = response.text  # Extracted response text
    
    # Store and display the assistant's response
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)