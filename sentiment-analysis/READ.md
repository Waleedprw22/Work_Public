

## Sentiment Analysis Project

### Overview
Using the source code from https://techvidvan.com/tutorials/python-sentiment-analysis/, I have implemented a wide variety of changes to perform Sentiment Analysis on airline tweets. One significant modification I made was transforming the problem from binary classification to multi-label classification. Initially, the source code only handled "Positive" or "Negative" labels, but I adjusted the data preprocessing to retain neutral sentiments as well. I used factorization and one-hot encoding to transform the sentiment labels into binary vectors for the multi-label classification task.

### Data Preprocessing
During data preprocessing, the sentiment labels were factorized to numerical form, and then one-hot encoding was applied to represent them as binary vectors. This allowed us to consider "Positive," "Negative," and "Neutral" sentiments as separate classes for the LSTM model.

### Model Architecture
The LSTM neural network was chosen for sentiment classification, which was transformed into a function to enable hyperparameter tuning. The model consists of an Embedding layer, SpatialDropout1D layer, LSTM layer with adjustable hyperparameters (units and dropout rate), and a Dense output layer with three neurons corresponding to the three sentiment classes (Positive, Negative, and Neutral). The activation function for the output layer was changed to categorical_crossentropy to accommodate multi-label classification.

### Hyperparameter Tuning
To find the best combination of hyperparameters for the LSTM model, a RandomizedSearchCV approach was implemented. The hyperparameter search space includes 'units' and 'dropout_rate.' The algorithm iterated through various combinations and retained the hyperparameters with the highest accuracy on the validation set.

### Training the Model
The model was trained with the best hyperparameters obtained from RandomizedSearchCV. The training process was conducted over five epochs, and adjustments to the epoch values were made to achieve higher training and validation accuracies (around 0.9 and 0.85, respectively).

### Sentiment Prediction
To evaluate the model, I generated 100 random test sentences with either positive or negative sentiments. The model was tested on these sentences, and the accuracy of the predictions was calculated and displayed. Prior to adjusting the model to multi-label classification, the accuracy of the model's performance on test data was .89.

### Results and Conclusion
With the multi-label classification setup and hyperparameter tuning, the sentiment analysis project achieved a test accuracy of approximately 0.84, demonstrating the effectiveness of the LSTM neural network in handling multi-label sentiment classification tasks. While GPUs could potentially improve model optimization, the current implementation showcases a practical application of machine learning in analyzing customer sentiments towards airline services.

In conclusion, the sentiment analysis project successfully adapted the source code to handle multi-label classification and utilized hyperparameter tuning for enhanced accuracy. Further improvements could be explored with larger and more diverse datasets, as well as additional optimization techniques with GPU support.

---

