import os
import gradio as gr
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Replace with your Hugging Face model repository identifier
MODEL_NAME = "Maxime-Bakunzi/medical-education-chatbot"

# Load model and tokenizer from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Set device: GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


def generate_response(question, model, tokenizer, max_length=600):
    """
    Generate a response from the model given a medical question.
    """
    # Prepend prefix to format the question as in training
    input_text = "answer medical question: " + question
    # Encode input and move to device
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate model output using beam search
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    # Decode the generated tokens to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def chatbot_interface(question):
    """
    Chatbot interface function for Gradio.
    """
    return generate_response(question, model, tokenizer)


# Create Gradio interface for the chatbot
demo = gr.Interface(
    fn=chatbot_interface,
    inputs=gr.Textbox(
        lines=2, placeholder="Ask a medical education question..."),
    outputs="text",
    title="Medical Education Chatbot",
    description="Ask questions about medical concepts, diseases, treatments, and more."
)

if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.launch(share=True)
