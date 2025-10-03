# SolarGuard: AI-Powered Solar Panel Defect Detection
➡️ [View the Live Application Here!](#)

> *(You can replace the placeholder above with a real screenshot of your app)*

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
