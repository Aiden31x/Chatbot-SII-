#!/bin/bash

# Create a virtual environment
echo "Creating virtual environment..."
python -m venv sii_chatbot_env

# Activate the virtual environment (Linux/Mac)
echo "Activating virtual environment..."
source sii_chatbot_env/bin/activate

# Install required packages
echo "Installing required packages..."
pip install gradio langchain faiss-cpu sentence-transformers langchain_community

# Install Ollama if not already installed
if ! command -v ollama &> /dev/null
then
    echo "Ollama not found. Please install Ollama from https://ollama.ai"
    echo "After installing, run: ollama pull mistral"
else
    echo "Checking for Mistral model in Ollama..."
    if ! ollama list | grep -q mistral
    then
        echo "Pulling Mistral model..."
        ollama pull mistral
    fi
fi

# Create necessary files
echo "Setting up Mistral prompt..."
ollama create mistral_prompt -f mistral_prompt.yaml

echo "Setup complete! Run the app with: python chatbot_app.py"