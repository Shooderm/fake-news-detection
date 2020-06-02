# fake-news-detection

## Data used: BuzzFeed and PolitiFact

Kaggle: https://www.kaggle.com/mdepak/fakenewsnet#BuzzFeed_fake_news_content.csv <br>
(kaggle上PolitiFact_fake_news_content.csv(825.42 KB)是错的。。是吧) <br>

## How to use
Go to `RNN-models` <br>
Run `Final Model`, which is our final model <br>
File `Visualization` has images about results of 4.1 Attention Visualization and 4.2 Bi-GRU Visualization

## Implementation details

For text preprocessing:
  1. do word tokenization, stopwords removal, lowercasing, lemmatization, stemming
  2. remove special objects, such as frequent words
  
For embedding layer:
  1. Word2Vec: skip-gram
  2. Doc2Vec: Lda2Vec
  
For the model:
  1. Bi-GRU
  2. Attention
  
For visualization:
  1. The clustering of LDA2Vec
  2. Attention score: different shades of red shown on the text
  3. Bi-GRU: shows how does these RNN model work on the content with blue and red colors shown on the text
 
## Software and Library Requirements
pytorch == 1.5.0+cu101 <br>
Keras == 2.3.1 <br>
matplotlib == 3.2.1 <br>
numpy == 1.18.4 <br>
pandas == 1.0.3 <br>
ntlk == 3.2.5 <br>
re == 2.2.1 <br>
spacy == 2.2.4 <br>
gensim == 3.6.0 <br>

Multicore-TSNE, tqdm <br>
