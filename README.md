# Emotion Detection Chatbot 🤖💬

A **Streamlit-based chatbot** that responds to user input while detecting and acknowledging **emotions and sentiment** using:

- **Hugging Face Transformers (`DialoGPT-medium`, `distilroberta-base`)** for chatbot response + emotion detection.
- **TextBlob** for sentiment analysis.
- **Streamlit** for an interactive web interface.

---

## 🚀 Features

✅ **Conversational chatbot** using `DialoGPT-medium`.  
✅ **Emotion detection** (`anger`, `joy`, `fear`, `sadness`, etc.) using `distilroberta-base`.  
✅ **Sentiment analysis** (`happy`, `sad`, `neutral`) using `TextBlob`.  
✅ Dynamic, clean **Streamlit interface**.  
✅ Modular, readable, and easy to extend.

---

## 🖥️ Demo

Run locally to chat with the bot, and it will:
- Generate relevant responses.
- Detect your emotion and sentiment.
- Provide empathetic replies based on your emotional state.

---


      
---

## 🛠️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/emotion-chatbot.git
cd emotion-chatbot
````
### 2️⃣ Set up a virtual environment (recommended)


```bash
python -m venv botenv
source botenv/bin/activate      # On Windows: botenv\Scripts\activate
````

---
### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
````
🚦 Usage
Run the chatbot locally with:

```bash

streamlit run app.py
````
The app will open in your default browser at:

```arduino

http://localhost:8501
````
---
##🧩 Requirements
All dependencies are listed in requirements.txt.

Key libraries used:

streamlit

transformers

torch

textblob

huggingface_hub

scikit-learn

nltk

pandas

numpy

---

##📊 Model Details
🤖 Chatbot
Model: microsoft/DialoGPT-medium
😄 Emotion Detection
Model: j-hartmann/emotion-english-distilroberta-base
😊 Sentiment Analysis
Library: TextBlob
---
🛠️ Customization
You can:
✅ Swap out DialoGPT with your preferred conversational model.
✅ Use multilingual/domain-specific models for emotion detection.

---
✍️ Author
👩‍💻 Nidhi Bajaj
---
📜 License
This project is open-source under the MIT License.

