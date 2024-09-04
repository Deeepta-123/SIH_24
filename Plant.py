import streamlit as st
import google.generativeai as genai

# Function to read API key from file
def read_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

# Set the path to your API key file
api_key_file = 'api_key.txt'

# Read the API key from the file
api_key = read_api_key(api_key_file)

# Configure the generative AI with the API key
genai.configure(api_key=api_key)

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# Streamlit UI
st.title("Chatbot")

# Accept user input for the prompt/message
prompt = st.text_input("Enter your message:")

# Function to generate response
def generate_response(prompt):
    convo = model.start_chat(history=[])
    convo.send_message(prompt)
    return convo.last.text

# Display chatbot response in a box
if prompt:
    with st.spinner('Generating response...'):
        response = generate_response(prompt)
    st.write("Chatbot:")
    st.info(response)
    st.write("\n")

# Ask if the chatbot can help with anything else
if st.button("Can I help you with anything else?"):
    prompt = st.text_input("Enter your message:")
    if prompt:
        with st.spinner('Generating response...'):
            response = generate_response(prompt)
        st.write("Chatbot:")
        st.info(response)
        st.write("\n")
