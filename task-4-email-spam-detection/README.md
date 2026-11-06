# SMS Spam Detection Using Machine Learning

A machine learning project that classifies SMS messages as **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP) and supervised learning algorithms.

This project was completed as part of the **Oasis Infobyte (OIBSIP) Data Science Internship – Task 4**.

## Project Overview

The objective of this project is to develop a machine learning model that can automatically identify unwanted SMS messages.

The project uses the SMS Spam Collection dataset and explores different classification algorithms to determine which model performs best at distinguishing spam messages from legitimate messages.

Special attention is given to spam recall because failing to detect spam messages is an important problem in spam filtering.

## Dataset

The project uses the **SMS Spam Collection Dataset**, containing SMS messages labeled as either spam or ham.

- Original dataset: 5,572 messages
- Dataset after removing duplicates: 5,169 messages
- Ham messages: 4,516 (87.37%)
- Spam messages: 653 (12.63%)

The dataset contains two columns:

| Column | Description |
|---|---|
| `label` | Message category: spam or ham |
| `message` | Original SMS message text |

The dataset is imbalanced, with significantly more legitimate messages than spam messages.

## Technologies and Libraries

- Python
- Jupyter Notebook
- Pandas – Data manipulation and analysis
- NumPy – Numerical operations
- Scikit-learn – Machine learning and model evaluation
- NLTK – Natural Language Processing
- Matplotlib – Data visualization
- Seaborn – Confusion matrix visualization
- Joblib – Model saving and loading

## Project Workflow

### 1. Data Loading and Exploration

- Loaded the SMS Spam Collection dataset.
- Inspected the dataset structure and sample messages.
- Checked for missing values and duplicate records.
- Removed duplicate messages.
- Analyzed the distribution of spam and ham messages.

### 2. Text Preprocessing

Prepared the SMS messages for machine learning by:

- Converting text to lowercase.
- Removing punctuation.
- Cleaning unnecessary characters.
- Applying tokenization to examine individual words.
- Removing English stopwords during TF-IDF vectorization.

### 3. Feature Extraction Using TF-IDF

Applied **Term Frequency–Inverse Document Frequency (TF-IDF)** to convert SMS text into numerical features that machine learning algorithms can understand.

TF-IDF assigns importance to words based on their frequency in a message and how common or rare they are across the entire collection of messages.

### 4. Train-Test Split

Divided the dataset into training and testing sets:

- Training data: 80% (4,135 messages)
- Testing data: 20% (1,034 messages)

Used stratified sampling to preserve the original spam and ham distribution in both sets.

### 5. Model Training

Trained and compared three classification models:

1. **Multinomial Naive Bayes** – A probabilistic classifier commonly used for text classification.
2. **Logistic Regression** – A supervised learning algorithm used for binary classification.
3. **Balanced Logistic Regression** – Logistic Regression with `class_weight='balanced'` to account for the imbalanced dataset.

The balanced Logistic Regression model was selected for the final implementation because it provided a stronger balance between detecting spam and correctly classifying legitimate messages.

### 6. Model Evaluation

Evaluated the models using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Spam recall received particular attention because it measures how many actual spam messages the model successfully identifies.

## Model Performance

### Initial Model Comparison

| Metric | Naive Bayes | Logistic Regression | Balanced Logistic Regression |
|---|---:|---:|---:|
| Accuracy | 94.87% | 96.03% | 97.10% |
| Spam Precision | 100% | 96% | 87% |
| Spam Recall | 60% | 72% | 91% |
| Spam F1-score | 75% | 82% | 89% |

The balanced Logistic Regression model achieved the highest initial test accuracy and significantly improved spam recall compared with the other models.

### Final Model Performance

The final implementation uses a Scikit-learn pipeline combining TF-IDF vectorization with balanced Logistic Regression.

| Metric | Final Test Score |
|---|---:|
| Accuracy | 96.71% |
| Spam Precision | 85% |
| Spam Recall | 89% |
| Spam F1-score | 87% |

### Final Confusion Matrix

```text
                 Predicted
                 Ham   Spam
Actual Ham        883    20
Actual Spam        14   117
```

Interpretation:

- 883 legitimate messages were correctly classified.
- 117 spam messages were correctly detected.
- 20 legitimate messages were incorrectly classified as spam.
- 14 spam messages were incorrectly classified as legitimate messages.

### 5-Fold Cross-Validation

To evaluate the consistency of the final pipeline, Stratified 5-Fold Cross-Validation was performed on the training dataset.

| Metric | Mean Score | Standard Deviation |
|---|---:|---:|
| Accuracy | 97.22% | 0.63% |
| Spam Precision | 89.86% | 2.97% |
| Spam Recall | 87.94% | 2.21% |
| Spam F1-score | 88.88% | 2.48% |

The cross-validation results indicate that the model maintains consistent performance across different training and validation splits.

## Why Spam Recall Matters

In spam detection, recall is particularly important because a false negative occurs when an actual spam message is classified as legitimate.

A low spam recall means more unwanted messages can pass through the filtering system.

The final model achieved **89% spam recall**, correctly identifying 117 out of 131 spam messages in the test dataset.

## Custom Message Predictions

The final trained model was also tested using new SMS messages that were not part of the original test dataset.

| Example SMS | Prediction |
|---|---|
| Congratulations! You have won a free iPhone. Claim your prize now! | Spam |
| Hey, are we still meeting tomorrow? | Ham |
| URGENT! You have been selected to receive 1000 dollars. Call now! | Spam |
| Can you please send me the assignment notes? | Ham |
| FREE entry into our weekly cash prize draw. Text WIN now! | Spam |
| I will be home in 20 minutes. | Ham |
| Your account has been selected for a special reward. Click here! | Spam |
| Good morning, hope you have a wonderful day. | Ham |

The predictions demonstrate that the model can distinguish common promotional and prize-related spam messages from ordinary conversations.

## Model Saving and Loading

The final trained pipeline was saved using Joblib:

```python
import joblib

joblib.dump(final_model, 'models/spam_detection_model.pkl')
```

The saved model can be loaded and reused without training it again:

```python
loaded_model = joblib.load('models/spam_detection_model.pkl')

message = ["Congratulations! You have won a free cash prize!"]

prediction = loaded_model.predict(message)

print("Prediction:", prediction[0])
```

Expected output:

```text
Prediction: spam
```

## Project Structure

```text
task-4-email-spam-detection/
│
├── data/
│   ├── SMSSpamCollection
│   └── sms_spam_collection.zip
│
├── models/
│   └── spam_detection_model.pkl
│
├── spam_detection.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/BeimnetTadesse/task-4-email-spam-detection.git
```

### 2. Navigate to the Project Directory

```bash
cd task-4-email-spam-detection
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**macOS / Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Jupyter Notebook

```bash
jupyter notebook
```

Open `spam_detection.ipynb` and run the notebook cells to explore the dataset, train the models, evaluate their performance, and test custom SMS messages.

## Conclusion

This project demonstrates how Natural Language Processing and machine learning can be used to build an SMS spam detection system.

Through data exploration, text preprocessing, TF-IDF feature extraction, model training, and evaluation, I developed a balanced Logistic Regression pipeline that achieved **96.71% test accuracy** and **89% spam recall**.

The project also provided practical experience with imbalanced datasets, model comparison, cross-validation, confusion matrix analysis, and saving trained machine learning models for future use.

## Author

**Beimnet Tadesse**

- GitHub: [BeimnetTadesse](https://github.com/BeimnetTadesse)
- Project Repository: [SMS Spam Detection](https://github.com/BeimnetTadesse/task-4-email-spam-detection)

**Internship:** Oasis Infobyte (OIBSIP) – Data Science Internship

**Task:** Task 4 – Email/SMS Spam Detection