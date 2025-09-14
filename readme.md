# Chat with Your Documents (PDF, PNG, JPG)

## Introduction

-----

**Chat with Your Documents** is a Python application that allows you to have a conversation with your documents, including PDF files and images (PNG, JPG, JPEG). You can ask questions in natural language, and the application will provide relevant answers based on the content of the uploaded files.

A key feature is the ability to choose between two modes of operation:

1.  **A private, local mode** using models that run directly on your computer.
2.  **A powerful cloud mode** using the Google Gemini API.

The application will only answer questions related to the content of the loaded documents.

## How It Works

-----

The application follows these steps to provide answers:

1.  **Loading and Processing:** The program reads the uploaded files. For PDFs with a text layer, it extracts the text directly. For scanned PDFs and images (PNG, JPG), it uses **Optical Character Recognition (OCR)** technology to convert the image into text.

2.  **Text Chunking:** All the extracted text is divided into small, logically related chunks for effective processing.

3.  **Embedding Generation:** The application uses a selected embedding model (either the local `instructor-xl` or a cloud-based model from Google) to convert the text chunks into numerical vectors that represent their semantic meaning.

4.  **Vector Storage and Retrieval:** The vectors are stored in a local vector database (FAISS). When you ask a question, it is also converted into a vector, and the database quickly finds the most semantically similar chunks from your documents.

5.  **Response Generation:** The retrieved chunks, along with your original question, are passed to a Large Language Model (either the local **Phi-3** or the cloud-based **Google Gemini**), which then generates a final, coherent answer.

## Installation and Setup

-----

### 1\. System Dependencies

Before installing the Python packages, you need to install system utilities for OCR and PDF processing.

  * **Tesseract-OCR** (for recognizing text from images):
    *For Debian/Ubuntu:*
    ```bash
    sudo apt update
    sudo apt install tesseract-ocr tesseract-ocr-eng
    ```
  * **Poppler** (for handling PDF files as images):
    *For Debian/Ubuntu:*
    ```bash
    sudo apt install poppler-utils
    ```

### 2\. Python Dependencies

1.  Clone the repository to your local machine.

2.  Download the library from the `requirements.txt` file.

3.  Install the dependencies by running the following command:

    ```bash
    pip install -r requirements.txt
    ```

### 3\. Model Configuration

You have two options:

  * **To use Google Gemini (Cloud Model):**

    1.  Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    2.  Create a `.env` file in the root directory of the project.
    3.  Add your key to the `.env` file in the following format:
        ```
        GOOGLE_API_KEY="YOUR_SECRET_API_KEY"
        ```

  * **To use the Local Model (Phi-3):**
    No API keys are needed. The models will be downloaded automatically on the first run. **Note:** This may take some time and requires a significant amount of disk space (several gigabytes).

## Usage

-----

1.  Ensure you have installed all system and Python dependencies. If you plan to use Gemini, add your API key to the `.env` file.

2.  Run the application using Streamlit. Execute the following command in your terminal:

    ```bash
    streamlit run app.py
    ```

3.  The application will open in your default web browser.

4.  In the sidebar, **first select the model provider** ("Google (Gemini)" or "Local (Phi-3)").

5.  Upload your documents (PDF, PNG, JPG, JPEG) and click the **"Process"** button.

6.  Once processing is complete, ask questions in the chat interface to get answers.