# MBTI-Tweetouilles

<p align="center"><b>Video showing embedding visualisation PCA/T-SNE before training</b></p>
<p align="center"><a href="https://youtu.be/sKwr3i8fq6g"><img src ="https://img.youtube.com/vi/sKwr3i8fq6g/0.jpg" /></a></p>

This project is meant to analyze user tweets, trying to determin users personality types according to MBTI types by analyzing the users tweets, it is also built on form of classes to be able to accept any data / categories so it can learn to classify many documents not only tweets.
The actual model can extend it's learning knowledge to new users / documents to categories classification it has never seen.

The actual program is based on two models:
- First one : Doc2Vec model (Neural Network)
- Second one : Logistic regression (Neural Network) on document embeddings

Work in progress :

- The doc2vec model has been trained, and need to be adjusted whith fine parameter tunning, actually the implementation of the bayesian optimization algorithm is under study area. [Test.py file]
- The logistic regression model is built and need adjustements on error tunning, manually analyzing missclassified labels could help.

Problems :

- The doc2vec model takes too much time to train so exploring the results latent space, takes too much time even with a bayesian optimization algorithm wich tries to minimize the number of steps needed to perform parameter tunning by evaluating the model's loss.
- The logistic regression model is based on doc_embeddings wich can be tricky to manually analyse, since it is hard to corelate manually users embeddings which are a bunch of number or tweets to the user personality and detecting the problem that can cause miss classification.
- Need for google cloud credits to continue learning, the project is transfered to local learning which can take long time even with small batch sizes, and fewer training examples.

Documents for reference and solution search :

- https://arxiv.org/pdf/1510.02675.pdf
- http://www.aclweb.org/anthology/W17-4780
