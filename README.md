# Medical Education Chatbot

Welcome to my Medical Education Chatbot project! This repository contains a domain-specific chatbot designed to provide medical education insights. The chatbot is built using a Transformer model fine-tuned on real-world medical dialogue data and is deployed via a user-friendly Gradio interface.

---

[DEMO VIDEO FROM Youtube](https://youtu.be/QNAzgu9f_cI)
 
[THE MODEL STORED IN HUGGING FACE](https://huggingface.co/Maxime-Bakunzi/medical-education-chatbot/tree/main)

[The Full DATASET FROM Hugging Face](https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot/blob/main/dialogues.parquet)
## Table of Contents

- [Medical Education Chatbot](#medical-education-chatbot)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Project Definition \& Domain Alignment](#project-definition--domain-alignment)
  - [Dataset Collection \& Preprocessing](#dataset-collection--preprocessing)
  - [Model Fine-Tuning \& Experiments](#model-fine-tuning--experiments)
  - [Evaluation Metrics](#evaluation-metrics)
  - [User Interface (Gradio)](#user-interface-gradio)
  - [Using the Hugging Face Model](#using-the-hugging-face-model)
  - [Repository Structure](#repository-structure)
  - [How to Run the Project](#how-to-run-the-project)
    - [Prerequisites](#prerequisites)
    - [Training the Model](#training-the-model)
    - [Running the Gradio Interface](#running-the-gradio-interface)
  - [Conclusion \& Future Work](#conclusion--future-work)
  - [Acknowledgements](#acknowledgements)

---

## Introduction

Hi there! In this project, I built a chatbot tailored for medical education using Transformer models. The goal was to create an interactive tool that delivers medically relevant responses to user queries, mimicking the experience of consulting a healthcare professional. This chatbot not only supports students and patients with quick advice but also demonstrates the application of advanced NLP techniques in the healthcare domain.

---

## Project Definition & Domain Alignment

**Purpose:**  
The chatbot is designed to answer questions related to medical concepts, diseases, treatments, and more. This specialized focus ensures that users receive responses that are medically informed rather than generic.

**Domain Relevance:**  
- **Medical Education:** Focuses on providing insights and explanations about various medical topics.
- **User Benefit:** It acts as an educational resource for patients, students, and healthcare professionals seeking immediate information.

---

## Dataset Collection & Preprocessing

**Dataset Source:**  
- The dataset was sourced from Hugging Face using the identifier `ruslanmv/ai-medical-chatbot`. This dataset includes dialogues between patients and doctors.

**Sampling & Storage:**  
- I sampled 10% of the full dataset for efficient training and saved it as a CSV file in the `dataset` directory.

**Preprocessing Steps:**  
- **Renaming:** The original columns were renamed to `instruction` (for patient queries) and `output` (for doctor responses).
- **Exploratory Analysis:** Basic statistics (such as average question and response lengths) were computed, and visualizations were generated to understand data distribution.
- **Tokenization:** The T5 tokenizer was used to convert text into token IDs. Each instruction was prefixed with `"answer medical question: "` and padded/truncated to a maximum of 512 tokens. Padding tokens in the labels were replaced with `-100` to ensure they do not affect the loss calculation.

---

## Model Fine-Tuning & Experiments

**Model Choice:**  
- I fine-tuned the `t5-small` model to handle medical dialogues efficiently without requiring extensive computational resources.

**Training Setup:**  
- **Hyperparameters:**  
  - Learning Rate: `5e-5`
  - Batch Size: `8`
  - Weight Decay: `0.01`
  - Number of Epochs: `3` (with early stopping based on validation loss)
- **Early Stopping:** To prevent overfitting, training was halted if the validation loss did not improve for two consecutive epochs.
- **Logging & Saving:** Training logs are stored in the `logs` directory, and the final model is saved in the `models` directory.

**Experiment Table:**

| Experiment | Epochs | Batch Size | Learning Rate | Training Loss | Validation Loss | BLEU Score  |
|------------|--------|------------|---------------|---------------|-----------------|-------------|
| 1          | 3      | 8          | 5e-5          | 3.688100      | 3.425483        | 0.001809    |
| 2          | 3      | 8          | 5e-5          | 3.529900      | 3.306601        | 0.001881    |
| 3          | 3      | 8          | 5e-5          | 3.563200      | 3.273671        | 0.003196    |

*Note: The BLEU scores indicate gradual improvements. Ongoing experiments aim to further enhance model performance through hyperparameter tuning and data augmentation.*

---

## Evaluation Metrics

The model was evaluated using:
- **BLEU Score:** Measures the overlap between generated responses and reference answers.
- **Validation Loss:** Used to monitor training progress.
- **Qualitative Analysis:** Sample queries were tested to ensure responses are medically sound and contextually relevant.

---

## User Interface (Gradio)

A Gradio interface was developed to allow users to interact with the chatbot in real time. The interface features:
- **Text Input:** Users can type in their medical questions.
- **Real-Time Response:** The model processes the query and returns an answer.
- **Public Access:** The interface can be launched with a public shareable link.

---

## Using the Hugging Face Model

I have uploaded the fine-tuned model to Hugging Face for easier deployment and sharing. The Gradio UI in `gradio_app.py` loads the model directly from Hugging Face, ensuring that anyone can access the same high-quality model without needing local checkpoints.

In the `gradio_app.py` file, replace the placeholder `MODEL_NAME` with your Hugging Face repository identifier (e.g., `"username/t5-medical-education"`). The code then downloads the model and tokenizer from the Hugging Face Hub and launches the Gradio interface with identical functionality as demonstrated in the training notebook.

---

## Repository Structure

```
medical-education-chatbot/
│
├── dataset/
│   └── medical_dialogues_sampled.csv      # Sampled dataset (10% of full data)
├── logs/
│   └── evaluation_results.txt             # Evaluation logs and training metrics
│
├── ui/
│   └── gradio_app.py                          # Standalone Gradio UI Python script (loads model from HF)
├── Medical_Education_Chatbot_Training.ipynb # Jupyter Notebook for training, evaluation, and experiments
└── README.md                              # This file
```

---

## How to Run the Project

### Prerequisites

- **Python 3.8+**
- Required packages: `datasets`, `transformers`, `torch`, `gradio`, `nltk`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

Install dependencies using:
```bash
pip install datasets transformers torch gradio nltk pandas scikit-learn matplotlib seaborn
```

### Training the Model

1. Open the Jupyter Notebook `Medical_Education_Chatbot_Training.ipynb`.
2. Follow the steps to load the dataset, preprocess the data, fine-tune the model, and evaluate performance.
3. The trained model and tokenizer will be saved in the `models` directory.

### Running the Gradio Interface

1. Ensure you have uploaded your fine-tuned model to Hugging Face.
2. In the `gradio_app.py` file, update `MODEL_NAME` with your Hugging Face model ID.
3. Run the Gradio UI by executing:
   ```bash
   python gradio_app.py
   ```
4. A public shareable link will be provided to interact with the chatbot.

---

## Conclusion & Future Work

This project has been a rewarding exploration into building a domain-specific chatbot for medical education. While the current model provides a strong foundation, further improvements can be made by:
- Expanding the dataset and exploring more advanced models (e.g., T5-base, T5-large).
- Refining hyperparameters and incorporating additional evaluation metrics such as F1 score and perplexity.
- Enhancing the UI for broader accessibility, including deployment on Hugging Face Spaces.

---

## Acknowledgements

I extend my sincere thanks to my instructors, peers, and the open-source community for their support. Special thanks to Hugging Face for their excellent tools and datasets that made this project possible.

---

Thank you for exploring my Medical Education Chatbot project!
