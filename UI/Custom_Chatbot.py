import streamlit as st
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from googletrans import Translator

# Load the intents.json file
with open("intents.json") as f:
    intents = json.load(f)

# Load the symptom classification model and tokenizer
model_name = st.secrets["repo"]  # Use your model name or path here
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

# Language code to full name mapping
language_mapping = {
    "en": "English", "hi": "Hindi", "mr": "Marathi", "bn": "Bengali", "pa": "Punjabi",
    "fr": "French", "es": "Spanish", "de": "German", "it": "Italian", "zh-cn": "Chinese"
}

# Initialize the translator
translator = Translator()

# Function to translate input to English and detect language
def translate_to_english(input_text):
    detection = translator.detect(input_text)
    detected_language_code = detection.lang
    detected_language_full = language_mapping.get(detected_language_code, detected_language_code)
    
    if detected_language_code != "en":
        translated = translator.translate(input_text, dest="en")
        return translated.text, detected_language_full
    return input_text, "English"

# Function to make predictions and get top symptoms with scores
def predict_top_symptoms(input_text, top_k=3, score_threshold=3):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = symptom_model(**inputs)
    
    logits = outputs.logits
    # Get the top k predictions
    top_values, top_indices = torch.topk(logits, top_k)
    
    symptoms_with_scores = []
    for i in range(top_k):
        symptom_index = top_indices[0][i].item()
        symptom_value = top_values[0][i].item()
        predicted_symptom = symptom_mapping.get(symptom_index, "Unknown symptom")
        
        # Only include symptoms with a score greater than or equal to the threshold
        if symptom_value >= score_threshold:
            symptoms_with_scores.append((predicted_symptom, symptom_value))
    
    return symptoms_with_scores

# Function to find responses from intents.json based on predicted symptoms
def get_responses(predicted_symptoms):
    responses = []
    for intent in intents["intents"]:
        if intent["tag"] in predicted_symptoms:
            responses.append(intent["responses"][0])  # Get the first response for simplicity
    return responses

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
    
    # Translate input to English and detect language
    translated_prompt, detected_lang = translate_to_english(prompt)
    
    # Predict the top symptoms
    top_symptoms_with_scores = predict_top_symptoms(translated_prompt)
    symptom_str = ", ".join([f"{symptom} (score: {score:.4f})" for symptom, score in top_symptoms_with_scores])
    
    # Get responses from intents.json based on the predicted symptoms
    predicted_symptoms = [symptom for symptom, _ in top_symptoms_with_scores]
    responses = get_responses(predicted_symptoms)
    
    if "Out of context" in predicted_symptoms:
        # Special response for "Out of context"
        response_msg = "I'm a medical chatbot relying on underlying symptom classification. Please input a general experience (I cannot work properly for diseases and non-medical prompts)."
    elif not responses:
        # If no medical-related symptom is predicted
        response_msg = "I am a medical chatbot, and I can assist you with health-related concerns. Please ask me a medical question."
    else:
        # Concatenate the responses for the top symptoms
        response_msg = "Here are some insights based on your symptoms:\n\n" + "\n\n".join(responses)
    
    # Include the predicted symptoms and their scores in the response
    if symptom_str:
        response_msg += f"\n\nTop Predicted Symptoms:\n{symptom_str}"
    else:
        response_msg += "\n\nNo significant symptoms detected."
    
    # Add the detected language to the response
    if detected_lang != "English":
        response_msg += f"\n\nNote: Your input was detected in {detected_lang} and translated to English."

    # Display the response
    st.session_state.messages.append({"role": "assistant", "content": response_msg})
    st.chat_message("assistant").write(response_msg)
