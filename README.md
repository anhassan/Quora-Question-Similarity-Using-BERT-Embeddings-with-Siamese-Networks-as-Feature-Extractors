# Quora Question Similarity Using BERT Embeddings with Siamese Networks as Feature Extractors
### By Ahmad Nayyar and Ganesh Rajasekar
### Custom Project- MSCI641 Electrical and Computer Engineering,University of Waterloo

### 1. Problem Statement

The problem that we would be focusing on this project is the Kaggle Competition of Quora
Question pairs. The task is to identify similar pair of questions as multiple questions with same
context and intent can cause major confusion in the Quora discussion forum.In doing so, the seekers
would not have to go through the hassle of searching the best answer among many and writers
would not have to write variant versions of the similar content.Quora currently uses Random Forest
to identify similar question but with the success of deep networks it is highly probable that they
can surpass the previous benchmarks.

### 2. Data

The data set contains 404351 question pair along with the label identifying each example as either
positive - duplicates or negative – non duplicates.There are 255045 negative (non-duplicate) and
149306 positive (duplicate) instances. This is a clear case of class imbalance. We in our analysis
would overcome this class imbalance by using both over sampling and under sampling in different
ways. The data set contains qid1,qid2 as the question descriptors, id as the question pair descriptor,
questions themselves and the label for each pair. Even though, the labels are subjective but we
will consider them as correct.As part of the Kaggle challenge rules, Kaggle has supplemented the
test set with computer-generated question pairs.
Dataset-https://www.kaggle.com/c/quora-question-pairs/data


### 3. Methodology and Results
#### a) Input Layer : Sentence Embeddings with different Word Embeddings

Why Required?

• Input required as a fixed length feature

• Bag of Words does not capture semantics

• Convert Pair of Questions to Vectors using Embeddings:

**Word Embeddings used:**

• Word2Vec (Predictive - using n-grams)

• Glove (Count based – using co-occurrence matrix)

• FastText (Faster – using Huffman algorithm while storing categories in the form of trees)

• BERT (Bi-directional -using Transformer’s Encoder) {used for feature engineering only}

**Sentence Embeddings used:**

• Sent2Vec (Unsupervised embeddings) with Word2Vec (E1)

• Sent2Vec (Unsupervised embeddings) with Glove (E2)

• InferSent (Supervised embeddings) with FastText (E3)

#### b) Feature Extraction Layer : Siamese Architectures with Sentence Features

**What are Siamese Networks?**

• Two or more identical subnetworks

• Find similarity or relationship between two comparable things(Question 1 & 2)

• Siamese Architectures used:

– 3 channel CNN Siamese (channel 1 with E1 embeddings , channel 2 with E2
embeddings , channel 3 with E3 embeddings)

– DNN Siamese 1 (with E1 embeddings)

– DNN Siamese 2 (with E2 embeddings)

– DNN Siamese 3 (with E3 embeddings)

**Refer to Poster.pdf and Siamese.jpg for more details**

Results and Ablation studies are displayed in poster as well.
### 4. Details:
 1. data.rar- Unzip data.rar to load the pre-trained embedding files
 2. main.py - Model Trained with proposed architecture giving best results
 3. BERT_embeddings- Script to get BERT Embeddings
 4. Feature_Engineering_BERT.py- 25 features generated using BERT Vectors thanks to Abhishek Thakur [4]
 5. get_sentence_embeddings.py - Glove,w2vec,fastext sentence embeddings using sent2vec and infersent
 6. models.py - Infersent pre-req file
 7. A Notebook Version of the Project is Also Available
 
### 5. Reference
[1] Ameya Godbole, Aman Dalmia, and Sunil Kumar Sahu. Siamese neural networks with
random forest for detecting duplicate question pairs. arXiv preprint arXiv:1801.07288, 2018.

[2] Yushi Homma, Stuart Sy, and Christopher Yeh. Detecting duplicate questions with deep
learning.

[3] Devlin, J., Chang, M.W., Lee, K. and Toutanova, K., 2018. Bert: Pre-training of deep
bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

[4] Quora Question Pair Duplicate Feature Engineering By Abhishek Thakur
https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur/
