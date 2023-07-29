Using the source code from https://techvidvan.com/tutorials/python-sentiment-analysis/, I have implemented a wide variety of changes. One big change I implemented was that
I turned the problem from a binary classification one to a multi label classification problem. Initially, the source code would take in data labeled as "Positive" or "Negative"
and train a text classifying model. I decided to adjust the data pre-processing chunk of code so that I can keep the neutral sentiments. The factorizing part stayed the same
but I used one hot encoding to transform the integer values for the labels into binary vectors. I also turned the model into a function so that I can conduct 
hyper tuning of the parameters. I adjusted the dense layer to have 3 neurons instead of one as I will have 3 outcomes: Positive, Negative or Neutral. I also had to change
the activation function to categorial_cross_entropy as I am dealing with multi label classification. The accuracy decreased from .96 to around .84 or so. I then made an algoritm
that can iterate through different hyper-parameters and find those that provide the highest accuracy and to retain those. I also played around with the epoch values and realized if
I increase it a bit, I can push the training accuracies to near .9 and val acuracies near .85. I then made an implementation that randomly generates 100 test sentences and test the
model with it, providing the user with test accuracies as well.

