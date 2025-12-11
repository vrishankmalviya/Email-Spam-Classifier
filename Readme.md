Email Spam Classifier
A machine learning project to classify emails as Spam or Ham (Not Spam) using text preprocessing, feature engineering, and Naive Bayes classifiers. The project also includes a Streamlit web app for interactive prediction.

1. Exploratory Data Analysis (EDA)
1.1 Text statistics per email

Computed the number of words, characters, and sentences in each email.

Used these features to understand length distributions and differences between spam and ham emails.

2. Data Preprocessing
Applied several NLP preprocessing steps to clean and normalize the text:

Lowercasing all text.

Tokenization using NLTK to split text into individual tokens.

Removing special characters to keep only alphanumeric tokens.

Removing stopwords and punctuation using NLTK’s English stopword list and Python’s string.punctuation.

Stemming using PorterStemmer to reduce words to their root form.

All preprocessing is encapsulated in a transform_text function (in preprocess.py) and reused in both the notebook (training) and the Streamlit app (inference).

3. Model Building
3.1 Feature extraction
Two main vectorization techniques were explored:

CountVectorizer: Converts text into a bag-of-words count representation.

TfidfVectorizer: Converts text into TF‑IDF features, giving more weight to informative words.

To improve model performance with TF‑IDF, different values of max_features were tested to control vocabulary size and reduce noise.

3.2 Classifiers
Three Naive Bayes variants were trained and evaluated:

GaussianNB

MultinomialNB

BernoulliNB

The goal was to identify which variant works best for sparse text data from Count/TF‑IDF features.

3.3 Evaluation
Since the dataset is not extremely large and misclassifying spam as ham is costly, the focus was on precision, along with standard metrics:

Accuracy score

Confusion matrix

Precision score

The best-performing model (in terms of precision on the validation/test set) was saved and integrated into the Streamlit app.

4. Tech Stack
Python

pandas, numpy

NLTK

scikit-learn

seaborn, matplotlib (for EDA and visualization)

Streamlit (for the web UI)

5. Project Structure
Example structure (may vary slightly from your repo):

text
.
├── app/
│   ├── web.py              # Streamlit app
│   ├── preprocess.py       # transform_text and text utilities
│   └── models/
│       ├── vectorizer.pkl
│       └── model.pkl
├── notebooks/
│   └── 01_spam_eda_and_model.ipynb
├── data/
│   └── README.md           # info / link to dataset
├── requirements.txt
├── README.md
└── .gitignore
6. Setup & Installation
bash
# Clone the repository
git clone https://github.com/<your-username>/email-spam-classifier.git
cd email-spam-classifier

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS / Linux

# Install dependencies
pip install -r requirements.txt
7. Running the Streamlit App
From the project root (or inside app/, depending on your structure):

bash
cd app
streamlit run web.py
Then open the provided local URL in your browser, enter an email/message in the text area, and the app will predict whether it is Spam or Ham using the trained model.

8. Notes
Training and evaluation code (EDA, feature engineering, and model comparison between Gaussian, Multinomial, and Bernoulli Naive Bayes) are in the Jupyter notebook under notebooks/.

The deployed Streamlit app uses the best-performing Naive Bayes model with TF‑IDF features tuned via max_features.