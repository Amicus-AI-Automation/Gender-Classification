# Gender Classification with Human Feedback Loop

This project builds a **machine learning system to predict gender from names** using **Logistic Regression**, enhanced with a **human-in-the-loop feedback mechanism** for continuous improvement.


## Project Overview

* Predicts **male / female / common** from a given name
* Uses **character-level text features** (n-grams, suffixes, vowels)
* Supports **confidence-based prediction**
* Learns continuously from **human feedback**
* Deployed using **Streamlit**
*  **PROJECT DOCUMENTATION** - [Project Documentation](https://docs.google.com/document/d/1QpheJPl9DeU1rAPYxFk3jy5S-Id0kxBoWUhVyMelG78/edit?tab=t.0)


## Workflow

1. Load raw name–gender dataset
2. Clean & normalize names (lowercasing)
3. Extract text features (n-grams, suffixes, vowels, length)
4. Vectorize features using `DictVectorizer`
5. Train Logistic Regression with class balancing
6. Validate using 5-fold cross validation
7. Save trained model and vectorizer
8. Predict gender with probability scores
9. Apply confidence threshold → mark uncertain cases as `common`
10. Collect human feedback (thumbs up/down + expert verdict)
11. Incrementally retrain model using feedback (warm start)
12. **FIGMA WORKFLOW** - [View Workflow Diagram](https://www.figma.com/board/G73G37tL3yiSzYNq2654x6/Gender-Classification-System?node-id=0-1&p=f&t=8lH7eAgOApmCUraj-0)

## Project Structure
```
.
├── model/
│   ├── gender_model.joblib        # Initial trained model (v1)
│   ├── gender_model_v2.joblib     # Feedback-updated model (v2)
│   └── vectorizer.joblib          # Feature vectorizer
│
├── feedback-identification.csv    # Human feedback storage
├── app.py                         # Streamlit application
├── train_model.ipynb              # Training & evaluation notebook
└── README.md
```


## How to Run

### Install dependencies

```bash
pip install pandas scikit-learn joblib streamlit
```

### Run Streamlit App

```bash
python -m streamlit run app.py
```


## Model Details

* **Algorithm:** Logistic Regression
* **Features:**

  * Character n-grams (2,3,4)
  * Prefix & suffix patterns
  * Vowel statistics
  * Name length
* **Evaluation:** Accuracy (CV + Test Set)
* **Class Handling:** Balanced weights

## Feedback & Continuous Learning

* User feedback is stored in a CSV file
* Correct predictions increase confidence weight
* Incorrect predictions trigger expert correction
* Model v2 is incrementally retrained using feedback

This creates a **self-improving ML system**.


## Tech Stack

* Python
* Scikit-learn
* Pandas
* Streamlit
* Joblib


## Use Cases

* Name-based gender inference
* NLP feature engineering practice
* Human-in-the-loop ML systems
* Real-world feedback-driven learning
