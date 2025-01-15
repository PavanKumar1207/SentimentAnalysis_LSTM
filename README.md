# Sentiment Analysis with LSTM and GloVe Word Embeddings

This project demonstrates a sentiment analysis system for restaurant reviews, built using an LSTM (Long Short-Term Memory) model and GloVe word embeddings (200-dimensional). The project includes a Flask-based web application to provide a user-friendly interface for analyzing the sentiment of textual reviews.

![Screenshot 2025-01-15 145805](https://github.com/user-attachments/assets/ea2745c3-3ab2-4c77-b8e2-95d8036f4705)


## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [How to Run the Application](#how-to-run-the-application)
- [Folder Structure](#folder-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project aims to classify restaurant reviews as either **positive** or **negative** using deep learning techniques. The application includes:
- Preprocessing of input text.
- An LSTM model trained with GloVe word embeddings for robust semantic understanding.
- A Flask web application for providing predictions via a user-friendly UI.

---

## Features

- **Sentiment Prediction**: Analyze reviews and classify them as positive or negative.
- **Preprocessing**: Text preprocessing to clean and standardize input text.
- **Interactive UI**: A simple web interface to input and analyze reviews.
- **Bootstrap Integration**: A responsive and visually appealing design.

---

## Technologies Used

- Python
- Flask
- TensorFlow/Keras
- GloVe Word Embeddings (200-dimensional)
- Bootstrap (UI Styling)

---

## Dataset

The model was trained on a dataset of restaurant reviews. The dataset was preprocessed to clean text and prepare it for training. Preprocessing steps included:
- Lowercasing text
- Removing special characters
- Tokenization and padding

---

## Model Training

- **Word Embeddings**: Used pre-trained GloVe embeddings (200 dimensions).
- **Neural Network**: LSTM layers for capturing temporal dependencies in text.
- **Training Parameters**:
  - Optimizer: Adam
  - Loss Function: Binary Crossentropy
  - Metrics: Accuracy

The trained model is saved as `sentiment_model.h5` for deployment.

---

## How to Run the Application

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or later
- TensorFlow
- Flask
- Numpy
- GloVe embeddings

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/PavanKumar1207/SentimentAnalysis_LSTM.git
   cd sentiment-analysis-lstm
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the trained model (sentiment_model.h5) and tokenizer (tokenizer.pkl) in the root directory.
  Run the Flask application:
  ```bash
  python app.py
  ```
5. Open your browser and go to http://127.0.0.1:5000.


# Folder Structure
```bash
sentiment-analysis-lstm/
│
├── app.py               # Flask application
├── preprocess.py           # Preprocessing utilities
├── Reviews.csv             # Dataset used to train
├── tokenizer.pkl           # Tokenizer for input text
├── sentiment_model.h5      # Trained LSTM model
├── templates/
│   └── index.html          # Frontend template
├── static/
│   ├── bg_1.jpeg           # Background image
├── requirements.txt        # Python dependencies
└── README.md  
```

# Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. **Fork the repository**.
2. **Create a new branch**:  
   ```bash
   git checkout -b feature-branch
3. Make your changes and commit them:
  ```bash
  git commit -m 'Add new feature'
  ```
4. Push to the branch:
  ```bash
  git push origin feature-branch
  ```
5. Open a pull request and describe your changes.

We appreciate all contributions, from bug fixes and feature enhancements to documentation improvements. Thank you for helping make this project better!

## License

This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute this software under the terms of the MIT License.

---

## Acknowledgments

- **GloVe Word Embeddings**: Used for vector representation of words in this project ([GloVe website](https://nlp.stanford.edu/projects/glove/)).
- **Bootstrap**: Utilized for styling the Flask web application's interface ([Bootstrap website](https://getbootstrap.com/)).
- **TensorFlow**: Employed for building and training the LSTM model ([TensorFlow website](https://www.tensorflow.org/)).

Special thanks to the open-source community for providing tools and resources that made this project possible.


