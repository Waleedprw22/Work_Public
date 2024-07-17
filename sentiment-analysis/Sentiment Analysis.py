import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D, Embedding
from tensorflow.keras.utils import to_categorical


# Data_pre-processing
df = pd.read_csv("./Tweets.csv")

tweet_df = df[['text','airline_sentiment']]
sentiment_label = tweet_df.airline_sentiment.factorize() #Convert words into numerical labels

tweet_df['airline_sentiment'] = sentiment_label[0]  #Replace original dataframe with the numerical sentiments.
sentiment_label = to_categorical(sentiment_label[0]) #converts numerical values into binary representation

# Tokenize the tweets
tweet = tweet_df.text.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(tweet)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)


embedding_vector_length = 50 #change from 32 to 50. Increased accuracy by a slight bit.

# Function to create the Keras model
def create_lstm_model(units=50, dropout_rate=0.2):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vector_length, input_length=200))
    model.add(SpatialDropout1D(0.25))
    model.add(LSTM(units, dropout=dropout_rate, recurrent_dropout=dropout_rate))
    model.add(Dense(3, activation='softmax'))  # Three output neurons for 3 classes
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Define hyperparameter search space
param_dist = {
    'units': [32, 64, 128],
    'dropout_rate': [0.2, 0.3, 0.4]
}

best_accuracy = 0.0
best_params = {}

# Loop through all hyperparameter combinations

for units in param_dist['units']:
    for dropout_rate in param_dist['dropout_rate']:
        print(f"Training with units={units}, dropout_rate={dropout_rate}")
        model = create_lstm_model(units=units, dropout_rate=dropout_rate)
        
        history = model.fit(padded_sequence, sentiment_label, validation_split=0.2, epochs=5, batch_size=32, verbose=0)
        val_accuracy = history.history['val_accuracy'][-1]
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = {'units': units, 'dropout_rate': dropout_rate}

print("Best Hyperparameters: ", best_params)

# Train the final model with the best hyperparameters
best_model = create_lstm_model(units=best_params['units'], dropout_rate=best_params['dropout_rate'])
best_model = create_lstm_model()
history = best_model.fit(padded_sequence, sentiment_label, validation_split=0.2, epochs=5, batch_size=32) 

plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
plt.savefig("Accuracy plot.jpg")


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.savefig("Loss plot.jpg")


def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw, maxlen=200)
    prediction_index = np.argmax(best_model.predict(tw))  # Use argmax to get the index of the predicted class
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    predicted_sentiment = sentiment_labels[prediction_index]

    return predicted_sentiment


# List of positive and negative words
positive_words = ["good", "excellent", "happy", "amazing", "wonderful", "fantastic", "love", "great", "delightful", "superb"]
negative_words = ["bad", "terrible", "horrible", "awful", "disappointing", "unpleasant", "poor", "miserable", "disgusting", "dislike"]


# Generate 100 random sentences with positive or negative labels
sentences = []
labels = []
for _ in range(100):
    if random.random() < 0.5:
        # Generate a positive sentence
        sentence = f"This {random.choice(positive_words)} experience is incredible!"
        label = "positive"
    else:
        # Generate a negative sentence
        sentence = f"The {random.choice(negative_words)} service was terrible."
        label = "negative"
    sentences.append(sentence)
    labels.append(label)

# Print the sentences and labels
count = 0
correct = 0

for i, (sentence, label) in enumerate(zip(sentences, labels), 1):
    count = count + 1
    print(f"{i}. {predict_sentiment(sentence)}  (Label: {label})")

    if str(predict_sentiment(sentence)) == str(label):
        correct = correct + 1
        
ratio = correct/count
print(f'The test accuracy is {ratio}')

