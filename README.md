# SolarGuard: AI-Powered Solar Panel Defect Detection
➡️ [View the Live Application Here!](#)

> *(https://solarpaneldefectdetection-7be8jqggtpdpc58dubxopu.streamlit.app/)*

---

## Project Overview
Solar energy is a critical renewable resource, but its efficiency is often hampered by environmental factors and physical damage. **SolarGuard** addresses this challenge by using deep learning to automatically classify the condition of solar panels from images.

Developed as a capstone for the **GUVI AI/ML course**, this project demonstrates an end-to-end machine learning workflow, from data analysis and model training to deployment as an interactive web application. The final model successfully classifies panels into six distinct categories with **84% accuracy**.

---

## Key Features
- **Image Classification:** Upload an image of a solar panel for instant analysis.
- **Six-Category Detection:** The model accurately classifies panels into:
  - Clean
  - Dusty
  - Bird Droppings
  - Electrical Damage
  - Physical Damage
  - Snow-Covered
- **Interactive Web App:** A user-friendly interface built with **Streamlit** makes the powerful AI model accessible to anyone.

---

## The Journey to High Accuracy
A key part of this project was the iterative model improvement process:

1. **Baseline Model:**  
   An initial training run yielded a low accuracy of ~20%, highlighting challenges with the imbalanced dataset.

2. **Fine-Tuning:**  
   By implementing transfer learning and fine-tuning a pre-trained **MobileNetV2** model, accuracy was significantly boosted to ~70%.

3. **Final Model (Class Weighting):**  
   The final model addressed the core problem of class imbalance by introducing class weights, which forced the model to pay closer attention to rarer defect types. This advanced technique pushed the final validation accuracy to an impressive **84%**.

---

## Technology Stack
- **Backend & Modeling:** Python, TensorFlow, Keras, Scikit-learn
- **Data Handling:** Pandas, NumPy
- **Frontend:** Streamlit
- **Image Processing:** Pillow, OpenCV

---

## Project Structure
.
├── dataset/ # Contains all the training and validation images
├── saved_model/
│ └── solar_panel_model_final.h5 # The final, trained Keras model
├── scripts/
│ └── app.py # The Streamlit application source code
├── model.ipynb # Jupyter Notebook with the full model training process
├── packages.txt # System-level dependencies for deployment
└── requirements.txt # Python library requirements


---

## Local Setup and Usage
To run this application on your local machine, follow these steps:

**Clone the repository:**
```bash
git clone https://github.com/jerw04/SolarPanel_Defect_Detection.git
cd SolarPanel_Defect_Detection
```
Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```
Install the required libraries:
```bash
pip install -r requirements.txt
```
Run the Streamlit app:
```bash
streamlit run scripts/app.py
```

