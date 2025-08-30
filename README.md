# 🐍 SerpAI - Snake Species Detection Model  

SerpAI is an **AI-powered snake species detection model** built with **TensorFlow** and deployed using **Streamlit**.  
It allows users to upload an image of a snake and instantly get the predicted **species name** along with whether the snake is **venomous or non-venomous**.  

---

## 📑 Table of Contents  

- [📖 Introduction](#-introduction)  
- [✨ Features](#-features)  
- [📂 Dataset](#-dataset)  
- [🚀 Installation](#-installation)  
- [⚙️ Setup](#️-setup)  
- [🖥️ Usage](#️-usage)  
- [🛠️ Tech Stack](#-tech-stack)  
- [📌 Requirements](#-requirements)  
- [🏗️ Project Structure](#️-project-structure)  
- [📄 License](#-license)  
- [🤝 Contributing](#-contributing)  
- [🙌 Acknowledgments](#-acknowledgments)  
- [📧 Contact](#-contact)  

---

## 📖 Introduction  

Snakebites are a significant public health issue, particularly in rural and tropical regions.  
Correct identification of snake species can save lives by ensuring the right medical treatment.  

**SerpAI** aims to:  
- Assist users, researchers, and medical professionals in identifying snake species.  
- Classify snakes into **venomous** or **non-venomous** categories.  
- Provide a lightweight, easy-to-use tool for rapid predictions.  

---

## ✨ Features  

- 🖼️ **Image Upload Support** → Accepts `.jpg`, `.jpeg`, `.png`.  
- 🤖 **Deep Learning Model** → TensorFlow CNN trained on snake species dataset.  
- ⚡ **Instant Predictions** → Displays species name with venomous/non-venomous tag.  
- 📊 **Streamlit Interface** → Simple, interactive web app for end users.  
- 🩺 **Life-Saving Insight** → Helps in early medical decision-making.  

---

## 📂 Dataset  

The dataset includes **multiple snake species** with labels stored in [`Snake_Species.csv`](./Snake_Species.csv).  

- **Columns include:**  
  - `Index` → Class label used by the model  
  - `Snake` → Name of the snake species  
  - `Category` → Venomous / Non-Venomous classification  

*(Sample snippet of dataset)*  

| Index | Snake                | Category      |  
|-------|----------------------|---------------|  
| 0     | Indian Cobra         | Venomous      |  
| 1     | Russell’s Viper      | Venomous      |  
| 7     | Rat Snake            | Non-Venomous  |  

---

## 🚀 Installation  

Clone the repo and install dependencies:  

```bash
git clone https://github.com/<your-username>/SerpAI-Snake-Detection.git
cd SerpAI-Snake-Detection
pip install -r requirements.txt
```
## ⚙️ Setup

1. Ensure you have the trained model file snake_classification.h5 in the project root. I have already trained the model so you can directly use it from my repo.
2. Ensure the dataset file Snake_Species.csv is present in the root directory. It's present in my repo.
3. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 🖥️ Usage

1. Launch the Streamlit app.
2. Upload a clear image of a snake (.jpg, .jpeg, .png).
3. Click Predict.
4. View the results:
   * Snake species name
   * Venomous / Non-Venomous classification

---

## 📓 Jupyter Notebooks

- basic_web_rag.ipynb → Shows how to perform RAG over a webpage using LangChain.
+ basic_wikipedia_rag.ipynb → Demonstrates RAG over Wikipedia content.

These notebooks provide a step-by-step breakdown of how RAG works without the Streamlit UI.

---

## 🛠️ Tech Stack

* Streamlit → Web UI for chatbot.
- TensorFlow → Deep learning framework for training & inference
+ Pandas → Handling species dataset
* Matplotlib → Data Visualization
- Numpy → Mathematical Computation
- Pillow (PIL) → Image preprocessing

---

## 📌 Requirements

See [requirements.txt](https://github.com/Mohd-Muzammil7052/SerpAI---A-Snake-Species-Detection-Model/blob/main/requirements.txt) for all dependencies:

```text
numpy == 1.26.4
tensorflow == 2.10.0
streamlit == 1.46.0
protobuf == 3.19.6
pandas == 1.5.3
scikit-learn == 1.0.2
matplotlib
```

---

## 🏗️ Project Structure  

```text
📦 SerpAI-Snake-Detection
 ┣ 📜 README.md                # Documentation
 ┣ 📜 app.py                   # Streamlit app for detection
 ┣ 📜 model.ipynb              # Notebook for training & experiments
 ┣ 📜 requirements.txt         # Dependencies
 ┣ 📜 Snake_Species.csv        # Snake species dataset
 ┗ 📜 snake_classification.h5  # Trained deep learning model
```

---

## 📄 License  

This project is licensed under the [MIT License](https://opensource.org/license/mit).  
Feel free to use, modify, and distribute it as needed.

---

## 🤝 Contributing  

Contributions are welcome! 🎉  
If you’d like to improve this project:  

1. Fork the repository  
2. Create a new branch (`git checkout -b feature-branch`)  
3. Commit your changes (`git commit -m "Add new feature"`)  
4. Push to the branch (`git push origin feature-branch`)  
5. Open a Pull Request  

---

## 🙌 Acknowledgments  

Special thanks to the amazing open-source tools powering this project:  

- [Tensorflow](https://www.tensorflow.org/)  
- [Numpy](https://numpy.org/)  
- [Pandas](https://pandas.pydata.org/)
- [Matplot](https://matplotlib.org/)
- [Streamlit](https://streamlit.io/)  

---

## 📧 Contact  

For queries or collaborations:  

**Mohd Muzammil**  
- [GitHub](https://github.com/Mohd-Muzammil7052)  
- [LinkedIn](https://www.linkedin.com/in/mohd-muzammil-109044290/)  

