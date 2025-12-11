# YouTube Shorts Engagement Optimizer 🎯  
A machine-learning project that predicts whether a YouTube Short will achieve high engagement based on its title and metadata.  
This repository includes full preprocessing, feature engineering, model training, evaluation, visualization, and model export (`.pkl`) for deployment.

---

## 📌 Project Overview
This project analyzes YouTube Shorts titles and performance metrics  
(view count, like count, comment count) to determine whether a Short is likely to receive **high engagement**.

The model uses a combination of:

- TF-IDF text vectorization  
- Clickbait pattern detection  
- Sentiment analysis (TextBlob)  
- Punctuation and title length features  
- Logistic Regression classifier  

It also exports a trained model + vectorizer for use in web apps, analytics dashboards, or automation pipelines.

---

## 🚀 Features
- Cleans and preprocesses real YouTube Shorts dataset
- Computes engagement score = (likes + comments) / views
- Creates binary target: high-engagement vs low-engagement
- Extracts meaningful textual and numeric features:
  - Title length
  - Punctuation (! ?)
  - Clickbait keyword detection
  - Sentiment polarity
- Uses **TF-IDF + Logistic Regression** for classification
- Achieves **~98% accuracy**
- Saves:
  - `logistic_model.pkl`
  - `tfidf_vectorizer.pkl`
  - `model_artifacts.zip`
- Includes multiple visualizations:
  - Title length distribution  
  - Clickbait vs engagement heatmap  
  - Confusion matrix  

---

## 📂 Repository Structure

    /dataset
        shorts_daily_stats_merged.csv
    /notebooks
       yt-shorts-engagement-optimizer.ipynb
    /models
        logistic_model.pkl
        tfidf_vectorizer.pkl
        model_artifacts.zip
   
    README.md

---

## 🧠 Model Pipeline (Step-by-Step)

### 1️⃣ Data Loading
Dataset is loaded from Kaggle input directory. Only required columns are retained:
- title
- viewCount
- likeCount
- commentCount

### 2️⃣ Target Variable Creation
Engagement ratio is calculated:

    engagement = (likeCount + commentCount) / viewCount

A median-based threshold splits titles into:
- 1 → high engagement  
- 0 → low engagement  

### 3️⃣ Feature Engineering
Engineered both text and numeric features:

    Title length  
    Punctuation count  
    Sentiment polarity  
    Clickbait detection (regex-based)  
    TF-IDF text features  

These features combined create a strong predictive model.

### 4️⃣ Model Training
A Logistic Regression model (max_iter=1000) is trained using:

- 3000 TF-IDF features  
- 4 numeric engineered features  

### 5️⃣ Evaluation
The model achieves:

    Accuracy: ~0.98
    Balanced precision/recall  
    Clear class separation  

### 6️⃣ Saving Artifacts
The following files are generated in `/kaggle/working`:

    logistic_model.pkl  
    tfidf_vectorizer.pkl  
    model_artifacts.zip  
    visualization PNG files  

---

## 📊 Visualizations Generated
- Distribution of title lengths  
- Clickbait vs high engagement (countplot)  
- Confusion matrix heatmap  

These help explain dataset behavior and model decisions.

---

## 💾 Saving the Model
The model is exported using:

    joblib.dump(model, "/kaggle/working/logistic_model.pkl")
    joblib.dump(tfidf, "/kaggle/working/tfidf_vectorizer.pkl")

Both artifacts are included in the ZIP as well.

---

## 🧪 Inference Example

    sample_title = "Top 10 tricks you won't believe - must see!"
    prediction = model.predict(vectorized_sample)

The repository also includes an extended inference function that returns:
- Predicted class  
- Probability of high engagement  
- Token-level contributions  
- Numeric feature contributions  

---

## 🎒 Technologies Used
- Python  
- Pandas / NumPy  
- Scikit-learn  
- TextBlob  
- Seaborn & Matplotlib  
- Joblib  
- Regex  
- Kaggle Notebook environment  

---

## 📈 Results Summary
- Extremely high prediction accuracy (~98%)  
- Strong separation of high vs low engagement  
- Clickbait, sentiment, and title structure significantly influence predictions  
- TF-IDF + engineered features outperform text-only models  

---

## 🔮 Future Improvements
- Train more complex models (XGBoost, LightGBM, Transformers)  
- Add SHAP-based explainability  
- Deploy as a Flask/Streamlit web app  
- Build a title generator using NLP  
- Add multi-label target (viral, average, low engagement)  

---

## 🧳 How to Run This Project

### 1️⃣ Clone the repository:

    git clone https://github.com/yourusername/youtube-shorts-engagement-optimizer.git
    cd yt-shorts-engagement-optimizer

### 2️⃣ Install required packages:

    pip install -r requirements.txt

### 3️⃣ Run the notebook or Python script:
(open in Jupyter or VS Code)

    yt-shorts-engagement-optimizer.ipynb

### 4️⃣ Use the saved `.pkl` files for inference or deployment.

---

## 🤝 Contributing
Feel free to open issues or submit pull requests for:
- Feature engineering ideas  
- Improved algorithms  
- Dataset expansion  

---

## 📄 License
This project is licensed under the MIT License.

---

## ⭐ Acknowledgements
- YouTube Data API  
- Kaggle Notebook environment  
- Scikit-learn documentation  
- Open-source community  

---

If you like this project, please ⭐ star the repository on GitHub!
