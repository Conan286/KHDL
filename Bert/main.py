import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the CSV file
data = pd.read_csv('sampled_reviews.csv')

# Combine Positive and Negative reviews into a single column
data['combined_review'] = data['Positive_Review'] + " " + data['Negative_Review']

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to convert reviews to BERT embeddings
def review_to_embedding(review):
    inputs = tokenizer(review, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()

# Apply BERT to all reviews and create vectors
data['review_vector'] = data['combined_review'].apply(review_to_embedding)

# Split the data into training (with ratings) and to predict (missing ratings)
train_data = data.dropna(subset=['Rating'])
missing_rating_data = data[data['Rating'].isna()]

# Prepare training data
X_train = np.vstack(train_data['review_vector'].values)
y_train = train_data['Rating'].values

# Train a linear regression model to predict ratings
model = LinearRegression()
model.fit(X_train, y_train)

# Predict ratings for rows with missing ratings
X_missing = np.vstack(missing_rating_data['review_vector'].values)
predicted_ratings = model.predict(X_missing)

# Ensure that predicted ratings are within [0, 10] by clipping
predicted_ratings = np.clip(predicted_ratings, 0, 10)

# Update the missing ratings with predicted values
data.loc[data['Rating'].isna(), 'Rating'] = predicted_ratings

# Save the updated data back to a CSV file
data.to_csv('output.csv', index=False)

print("Predicted ratings saved to 'predicted_ratings_with_bert.csv'")
