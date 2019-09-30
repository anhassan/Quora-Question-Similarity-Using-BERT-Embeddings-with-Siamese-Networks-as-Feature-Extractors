# The Sentence Embeddings are saved as files to get your own Sentence Embeddings Uncomment and use this below functions
#Cmd Functions:
# !pip install gensim
# !pip install git+https://github.com/oborchers/Fast_Sentence_Embeddings
# !mkdir GloVe
# !curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
# !unzip GloVe/glove.840B.300d.zip -d GloVe/
# !mkdir encoder
# !curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl

#Libraries Needed
from bert_serving.client import BertClient
import gensim
from gensim.models import Word2Vec, FastText
from fse.models import Sentence2Vec
# Make sure, that the fast version of fse is available!
from fse.models.sentence2vec import CY_ROUTINES
assert CY_ROUTINES
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from models import InferSent
# -i/p needed for functions: 
# a)Sentences_tok - Tokenized Sentences
# b) sentences- Sentences in list of list

def infersent_glove():
    #Set Model for InferSent+Glove
    V = 1
    MODEL_PATH = '/tmp/GloVe/encoder/infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    modelg = InferSent(params_model)
    modelg.load_state_dict(torch.load(MODEL_PATH))
    # Keep it on CPU or put it on GPU
    use_cuda = True
    modelg = modelg.cuda() if use_cuda else modelg

    # If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
    W2V_PATH = '/tmp/GloVe/glove.840B.300d.txt' if V == 1 else '/home/ganesh/Quora_dev/tmp/GloVe/glove.840B.300d.txt'
    modelg.set_w2v_path(W2V_PATH)
    # Load embeddings of K most frequent words
    modelg.build_vocab_k_words(K=100000)
    return modelg

def get_fastext(sentences_tok):
  print("Training FastText model ...\n")
  model = FastText(size=324, window=10, min_count=1)  # instantiate
  model.build_vocab(sentences_tok)
  model.train(sentences=sentences_tok,total_examples=len(sentences_tok),epochs=5)  # train
  se = Sentence2Vec(model)
  ft_embeddings = se.train(sentences_tok)
  return ft_embeddings
  

def get_w2v(sentences_tok):
  #Word2Vec Embeddings
  print("Training W2v model ...\n")
  w2v_model = Word2Vec(sentences_tok,size=324,window=10, min_count=1)
  se = Sentence2Vec(w2v_model)
  w2v_embeddings = se.train(sentences_tok)
  return w2v_embeddings

def get_glove(sentences):
  print("Training glove+infersent model ...\n")
  embeddings = modelg.encode(sentences, bsize=128, tokenize=False, verbose=True)
  pca = PCA(n_components=324) #reduce down to 50 dim
  glove_embeddings = pca.fit_transform(embeddings)
  return glove_embeddings

def get_bertembeddings(q1,q2):
  q = list(map(lambda x, y: x+ ' ||| ' +y, q1, q2))
  bc = BertClient()
  bert_embeddings = bc.encode(q)
  return bert_embeddings

modelg = infersent_glove()



df_sub = pd.read_csv('data/data.csv')
print('Shape of Dataset',df_sub.shape)


df_sub['question1'] = df_sub['question1'].apply(lambda x: str(x))
df_sub['question2'] = df_sub['question2'].apply(lambda x: str(x))
q1sents = list(df_sub['question1'])
q2sents = list(df_sub['question2'])
tokenized_q1sents = [word_tokenize(i) for i in list(df_sub['question1'])]
tokenized_q2sents = [word_tokenize(i) for i in list(df_sub['question2'])]

#Get Fastext Sentence Embeddings
ft_emb_q1 = get_fastext(tokenized_q1sents)

#Get w2vec-sent2vec Sentence Embeddings
w2v_emb_q1 = get_w2v(tokenized_q1sents)

#Get Glove Sent2vec Sentence Embeddings
glove_emb_q1 = get_glove(q1sents)


#Compute Embeddings for Q2 Pair
ft_emb_q2 = get_fastext(tokenized_q2sents)

w2v_emb_q2 = get_w2v(tokenized_q2sents)

glove_emb_q2 = get_glove(q2sents)


#Save all the other arrays in CSV and use Main.py in this format
glove_emb_q1.to_csv('./data/Embeddings/glove/glove_q1_balanced.csv', index=False)