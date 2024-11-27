# Sentiment Analysis for E-Commerce (Naive Bayes)

This project performs sentiment analysis on e-commerce reviews using a Naive Bayes classifier to classify sentiments as positive or negative.

## Steps

1. **Install Required Libraries**  
   Install the necessary libraries: `pandas`, `matplotlib`, `Sastrawi`, `wordcloud`, `numpy`, and `scikit-learn`.

2. **Import Libraries**  
   Import the required Python libraries for data handling, visualization, text processing, and machine learning.

3. **Data Crawling**  
   Gather e-commerce review data (either via scraping or from a dataset) and load it into a Pandas DataFrame.

4. **Data Preprocessing**  
   Clean the data by removing duplicates, handling missing values, and normalizing the text (e.g., converting to lowercase).

5. **Feature Extraction**  
   Convert the review text to numerical data using TF-IDF (Term Frequency-Inverse Document Frequency).

6. **Train Naive Bayes Classifier**  
   Split the data into training and testing sets. Train the Naive Bayes classifier on the training set and predict sentiment on the test set.

7. **Model Evaluation**  
   Evaluate model performance using metrics such as:
   - **Accuracy**: Proportion of correct predictions.
   - **Precision, Recall, F1-score**: Detailed metrics to assess performance.

8. **Visualization**  
   Generate visualizations like:
   - **Word Cloud**: Show the most frequent words in the reviews.
   - **Sentiment Distribution**: Visualize the distribution of sentiments (positive, neutral, negative).

---

## Outputs and Results

- **Model Metrics**: Evaluate the model's accuracy, precision, recall, and F1-score.
- **Visualizations**:
   - **Word Cloud**: Highlights frequent words tied to positive/negative sentiments.
   - **Sentiment Distribution**: Displays the breakdown of sentiments in the dataset.
- **Model Comparison**: Compare the Naive Bayes classifier to baseline models (e.g., random classifier) to assess its effectiveness.

---

## Additional Notes

- Ensure all dependencies are installed before running the code.
- The dataset should include columns like `review` (review text) and `sentiment` (sentiment label: positive, negative, neutral).
- Further improvements can be made with hyperparameter tuning or using more advanced NLP techniques.
