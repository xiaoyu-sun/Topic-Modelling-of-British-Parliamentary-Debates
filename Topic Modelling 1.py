""""###########################################################
PART 1: Data Processing
https://radimrehurek.com/gensim/models/ldamulticore.html
############################################################"""

import math
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split
import numpy as np
import random

import nltk
nltk.download('wordnet')
from nltk.stem import *
import matplotlib.pyplot as plt

from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel



"""
DATA PROCESSING Functions
"""
def lemmatize_stemming(text):
    return SnowballStemmer("english").stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if len(token)>2 and (token not in gensim.parsing.preprocessing.STOPWORDS):
            result.append(lemmatize_stemming(token))
    return result


"""
DOCUMENT C --- 1990-1999
processed_documentC, dictionaryC, bow_corpusC, corpus_tfidfC
bow_corpusC and corpus_tfidfC further split into test and train set
"""

documentC = pd.DataFrame({'index': list(range(13206)), 'text': ''})
for i in range(13206):
    with open("/Users/xiaoyusun/Downloads/TTIC 31220/Project/data2/C"+str(i+1)+".txt", 'r') as file:
        documentC.loc[i,'text'] += file.read().replace('\n', ' ')
        
# Preprocess the documents
processed_documentC = documentC['text'].map(preprocess)
# Create a dictionary containing every unique word from all the documents
dictionaryC = gensim.corpora.Dictionary(processed_documentC)
# see the length of dictionary (71863)
len(dictionaryC.keys())

# Filter out tokens that appear in fewer than 5 documents or more than 50% of all the documents
dictionaryC.filter_extremes(no_below=5, no_above=0.5)
# see the length again (26456)
len(dictionaryC.keys())

# For each document, we create a dictionary recording what tokens it contains, and how many.
# Save all these dictionaries in a list.
bow_corpusC = [dictionaryC.doc2bow(doc) for doc in processed_documentC]
tfidfC = models.TfidfModel(bow_corpusC)
corpus_tfidfC = tfidfC[bow_corpusC]
#Is a D(=13206) length vector, inside is n unique words of element, each element is (word, td-idf)

df_tfidfC = pd.DataFrame(data=0, index=list(range(13206)), columns=[dictionaryC[i] for i in range(len(dictionaryC))])
array_tfidfC = np.array(df_tfidfC, dtype="float")

for i in range(13206):
    if i % 3000 == 0:
        print(i, "documents recorded...")
    for j,k in corpus_tfidfC[i]:
        array_tfidfC[i,j] = k
print("Done!")

df_tfidfC = pd.DataFrame(data = array_tfidfC, index=list(range(13206)), columns=[dictionaryC[i] for i in range(len(dictionaryC))])

"""
Split dataset into training and validation sets
"""
lst = list(range(len(documentC)))
np.random.seed(2019)
test_index = random.sample(lst, round(len(documentC)/3))

test_corpusC = []
for number in test_index:
    value = bow_corpusC[number]
    test_corpusC.append(value)
train_corpusC = np.delete(bow_corpusC, test_index).tolist()

test_corpus_tfidfC = tfidfC[test_corpusC]
train_corpus_tfidfC = tfidfC[train_corpusC]

# see the whole vocabulary
words = [dictionaryC[i] for i in range(len(dictionaryC))]
sorted(words)[2000:2100]


    
"""
DOCUMENT B --- 2009-2018
processed_documentB, dictionaryB, bow_corpusB, corpus_tfidfB
bow_corpusB and corpus_tfidfB further split into test and train set
"""
documentB= pd.DataFrame({'index': list(range(9231)), 'text': ''})
for i in range(9231):
    with open("/Users/xiaoyusun/Downloads/TTIC 31220/Project/data2/B"+str(i+1)+".txt", 'r') as file:
        documentB.loc[i,'text'] += file.read().replace('\n', ' ')

# Preprocess the documents from 2009-2018:
processed_documentB = documentB['text'].map(preprocess)

# Create a dictionary containing every unique word from all the documents:
dictionaryB = gensim.corpora.Dictionary(processed_documentB)
# Number of unique words in all the documents
len(dictionaryB)
# Filter out tokens that appear in fewer than 5 documents or more than 50% of all the documents:
dictionaryB.filter_extremes(no_below=5, no_above=0.5)
len(dictionaryB)  # 26752

# DID NOT RUN THIS PART ---------------------------
word_list = []
for k, words in dictionaryB.iteritems():
    word_list.append(words)

# For each document, we create a dictionary recording what tokens it contains, and how many.
# Save all these dictionaries in a list:
bow_corpusB = [dictionaryB.doc2bow(doc) for doc in processed_documentB]

tfidfB = models.TfidfModel(bow_corpusB)
corpus_tfidfB = tfidfB[bow_corpusB]
#Is a D length vector, inside is n unique words of element, each element is (word, td-idf)

df_tfidfB = pd.DataFrame(data=0, index=list(range(9231)), columns=[dictionaryB[i] for i in range(len(dictionaryB))])
array_tfidfB = np.array(df_tfidfB, dtype="float")
for i in range(9231):
    if i % 3000 == 0:
        print(i, "documents recorded...")
    for j,k in corpus_tfidfB[i]:
        array_tfidfB[i,j] = k
print("Done!")
df_tfidfB = pd.DataFrame(data = array_tfidfB, index=list(range(9231)), columns=[dictionaryB[i] for i in range(len(dictionaryB))])

"""
Split dataset into training and testing
"""
#SPLIT INTO TRAINING AND VALIDATION
lst = list(range(len(documentB)))
np.random.seed(2019)
test_index = random.sample(lst, round(len(documentB)/3))

test_corpusB = []
for number in test_index:
    value = bow_corpusB[number]
    test_corpusB.append(value)
train_corpusB = np.delete(bow_corpusB, test_index).tolist()

test_corpus_tfidfB = tfidfB[test_corpusB]
train_corpus_tfidfB = tfidfB[train_corpusB]






""""###########################################################
TOPIC MODELLING
############################################################"""

"""
Functions relevant for all Models
"""
#Given model, return T x W matrix
def get_topic_word_mat(model):
    return model.get_topics()
    
#Given model, n_topics, n_docs, return D X T matrix
def get_doc_topic_mat(lda_model, n_topics, corpus):
    document_mat = np.zeros((len(corpus), n_topics))
    for doc_idx in range(len(corpus)):
        current_doc = corpus[doc_idx]
        for index, score in lda_model[current_doc]:
            document_mat[doc_idx][index] = score
    return document_mat

    
def plot_coherence_values(ModelType, dictionary = None, corpus=None, test = None, text=None,
                             stop=10, start=2, step=3, coherence_type = 'c_uci'):
    """
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for n_topics in range(start, stop, step):
        # generate model
        model_coherence = ModelType(corpus=corpus, num_topics=n_topics, 
                          id2word=dictionary)  # train model
        model_list.append(model_coherence)
        coherencemodel = CoherenceModel(model=model_coherence, corpus= test, texts = text,
                                        dictionary=dictionary, coherence=coherence_type)
        coherence_values.append(coherencemodel.get_coherence())
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    
"""
Perform Latent Dirichlet Allocation
DOCUMENT C
"""
#Perform tuning based on Coherence Score
plot_coherence_values(gensim.models.LdaMulticore, dictionary = dictionaryC, 
                      corpus=train_corpus_tfidfC, test=None, text = processed_documentC[test_index],
                      stop=40, start=1, step=3)


#Initiate model
#Now, we just fix a number for topic C
#Let eta be automatic for now
n_topics = 16
lda_model_C = gensim.models.LdaMulticore(train_corpus_tfidfC, num_topics=n_topics, id2word=dictionaryC, 
                                       alpha=1, eta='auto', passes=7, workers=4)

#Ordered by significance
for idx, topic in lda_model_C.print_topics(num_topics=-1, num_words=20):
    print('Topic: {} \nWords: {}'.format(idx, topic))

#Get TxW matrix
TxW_mat = get_topic_word_mat(lda_model_C)
DxT_mat = get_doc_topic_mat(lda_model_C, n_topics, train_corpus_tfidfC)

#Can also use this to show topics
#Only reveal words, not scores
topic_list = []
for topic_id, topic in lda_model_C.show_topics(num_topics=n_topics, formatted=False):
    topic = [word for word, _ in topic]
    topic_list.append(topic)


#Topic probability: Score = probability that topic is in document
sample_doc = train_corpus_tfidfC[1]

total_score=0
for index, score in sorted(lda_model_C[sample_doc], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, 
          lda_model_C.print_topic(index, 10)))
    total_score = total_score + score

#returns T x W matrix
TxW_mat = lda_model_C.get_topics()

#Return words for a given topic
lda_model_C.print_topic(0, 30)

#returns vector of probability for a given topic
lda_model_C.get_topic_terms(topicid=2, topn=20)
#get top topics based on model coherence score
lda_model_C.top_topics(corpus=train_corpus_tfidfC, texts=processed_documentC[test_index], 
                       dictionary=dictionaryC, window_size=None, coherence='c_uci', topn=20)


#TEST ON UNSEEN DOCUMENTS
#Output: topic probability for documents
unseen_document = documentC.iloc[1]
bow_vector = dictionaryC.doc2bow(preprocess(unseen_document))
for index, score in sorted(lda_model_C[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_C.print_topic(index, 5)))

#get_document_topics(bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)

"""
Get Coherence Score
Other functions in CoherenceModel: compare_models(models)
"""
cm1 = CoherenceModel(model = lda_model_C, corpus = train_corpus_tfidfC, 
                     dictionary=dictionaryC, coherence='u_mass')
cm1.get_coherence()
# tuning!
n_topics = 7
lda_model_C = gensim.models.LdaMulticore(train_corpus_tfidfC, num_topics=n_topics, id2word=dictionaryC, 
                                       alpha=0.2, eta='auto', passes=2, workers=4)
cm2 = CoherenceModel(model = lda_model_C,
                          texts = processed_documentC[test_index], dictionary=dictionaryC, 
                          coherence='c_uci')
cm2.get_coherence()

# After tuning, we choose alpha=0.5:
n_topics = 16
lda_model_C = gensim.models.LdaMulticore(corpus_tfidfC, num_topics=n_topics, id2word=dictionaryC, 
                                       alpha=0.5, eta='auto', passes=7, workers=4)
cm2 = CoherenceModel(model = lda_model_C,
                          texts = processed_documentC, dictionary=dictionaryC, 
                          coherence='c_uci')
cm2.get_coherence()
for idx, topic in lda_model_C.print_topics(num_topics=-1, num_words=20):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# see the topics for T=7:
lda_model_C1 = gensim.models.LdaMulticore(corpus_tfidfC, num_topics=7, id2word=dictionaryC, 
                                       alpha=0.3, eta='auto', passes=7, workers=4)
cm1 = CoherenceModel(model = lda_model_C1,
                          texts = processed_documentC, dictionary=dictionaryC, 
                          coherence='c_uci')
cm1.get_coherence()
for idx, topic in lda_model_C1.print_topics(num_topics=-1, num_words=20):
    print('Topic: {} \nWords: {}'.format(idx, topic))
    
DxT_mat_lda7 = get_doc_topic_mat(lda_model_C1, 7, corpus_tfidfC)
DxT_mat_lda16 = get_doc_topic_mat(lda_model_C, 16, corpus_tfidfC)
TxW_mat_lda7 = get_topic_word_mat(lda_model_C1)
"""
Perform Latent Semantic Analysis (Baseline Method)
Document C
"""
from gensim.models import LsiModel

#Perform tuning based on Coherence Score
plot_coherence_values(LsiModel, dictionary = dictionaryC, 
                      corpus=train_corpus_tfidfC, test=None, text = processed_documentC[test_index],
                      stop=40, start=1, step=3)


lsa_model_C = LsiModel(train_corpus_tfidfC, num_topics=7, 
                    id2word = dictionaryC)  # train model

for idx, topic in lsa_model_C.print_topics(num_topics=-1, num_words=10):
    print('Topic: {} \nWords: {}'.format(idx, topic))

cm_lsa = CoherenceModel(model = lsa_model_C, corpus = test_corpus_tfidfC,
                        texts = processed_documentC[test_index],
                        dictionary=dictionaryC, coherence='c_uci')
cm_lsa.get_coherence()

#tune
lsa_model_C1 = LsiModel(train_corpus_tfidfC, num_topics=7, 
                    id2word = dictionaryC)
cm_lsa1 = CoherenceModel(model = lsa_model_C1, corpus = test_corpus_tfidfC,
                        texts = processed_documentC[test_index],
                        dictionary=dictionaryC, coherence='c_uci')
cm_lsa1.get_coherence()

# whole dataset
lsa_model_C = LsiModel(corpus_tfidfC, num_topics=7, 
                    id2word = dictionaryC)
cm_lsa = CoherenceModel(model = lsa_model_C, corpus = corpus_tfidfC,
                        texts = processed_documentC,
                        dictionary=dictionaryC, coherence='c_uci')
cm_lsa.get_coherence()




"""
Hierarchical Dirichlet Processes
https://radimrehurek.com/gensim/models/hdpmodel.html
"""
from gensim.models import HdpModel

hdp_model_C = HdpModel(corpus=train_corpus_tfidfC, id2word=dictionaryC,  
                       alpha=0.5, gamma=1, eta=0.01, 
         outputdir=None, random_state=None)

hdp_model_C.show_topic(topic_id=0)
#Rough look at topics
for idx, topic in hdp_model_C.print_topics(num_topics=10, num_words=20):
    print('Topic: {} \nWords: {}'.format(idx, topic))

#Return words for a given topic
hdp_model_C.print_topic(149, 30)

    
#Find list of topics
hdptopics = hdp_model_C.show_topics(num_topics = 20, formatted=False)
hdptopics = [[word for word, prob in topic] for topicid, topic in hdptopics]

#Can also use this to show topics
#Only obtain words
topic_list_hdp = []
hdp_model_C.show_topics(num_topics=-1)
for topic_id, topic in hdp_model_C.show_topics(num_topics=-1, formatted=False):
    topic = [word for word, _ in topic]
    topic_list.append(topic)

#Print topics for each document
sample_doc = train_corpus_tfidfC[10200]
for index, score in sorted(hdp_model_C[sample_doc], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, 
          hdp_model_C.print_topic(index, 10)))

    
#Evaluate the coherence score
CoherenceModel(topics=hdptopics, texts=processed_documentC[test_index], 
               dictionary=dictionaryC, window_size=10).get_coherence()

#TUNE ALPHA
coherence_values = []
model_list = []
x = [0.01, 0.05, 0.1, 0.5, 1, 5]
log_x = [math.log(num) for num in x]
for alpha_val in x:
        # generate model
        model_coherence = HdpModel(train_corpus_tfidfC, id2word=dictionaryC,  alpha=alpha_val, gamma=1, eta=0.01, 
         outputdir=None, random_state=None)  # train model
        model_list.append(model_coherence)
        hdptopics = model_coherence.show_topics(num_topics = 10, formatted=False)
        hdptopics = [[word for word, prob in topic] for topicid, topic in hdptopics]
        coherence_val = CoherenceModel(topics=hdptopics, texts=processed_documentC[test_index], 
               dictionary=dictionaryC, coherence='u_mass', window_size=10).get_coherence()
        coherence_values.append(coherence_val)
plt.plot(log_x, coherence_values)
plt.xlabel("Log(alpha)")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# New

coherence_values = []
x = [0.005, 0.01, 0.05, 0.1, 0.5, 1]
log_x = [math.log(num) for num in x]
for alpha_val in x:
        # generate model
        model_coherence = HdpModel(train_corpus_tfidfC, id2word=dictionaryC,  
                                   alpha=0.1, eta= alpha_val,
         outputdir=None, random_state=None)  # train model
        hdptopics = model_coherence.show_topics(formatted=False)
        hdptopics = [[word for word, prob in topic] for topicid, topic in hdptopics]
        coherence_val = CoherenceModel(topics=hdptopics[:10], texts=processed_documentC[test_index], 
               dictionary=dictionaryC, coherence='u_mass', window_size=10, ).get_coherence()
        coherence_values.append(coherence_val)
        
y = [num/10 for num in coherence_values]
plt.plot(log_x, y)
plt.xlabel("Log(eta)")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


#Testing-------------------
model_coherence = HdpModel(train_corpus_tfidfC, id2word=dictionaryC,  
                                   alpha=0.1, eta=0.1, outputdir=None, random_state=None)  # train model
hdptopics = model_coherence.show_topics(formatted=False)
hdptopics = [[word for word, prob in topic] for topicid, topic in hdptopics]
coherence_val = CoherenceModel(topics=hdptopics[:10], texts=processed_documentC[test_index], 
        dictionary=dictionaryC, coherence='u_mass', window_size=10, ).get_coherence()
print(coherence_val)
""""###########################################################
Visualization
############################################################"""
from utils import LowEmb, tsne, plot_embedding
EPS=1.0e-12
torch.manual_seed(1)
from sklearn.manifold import TSNE

"""
t-SNE Dimensionality Reduction
"""

#First perform on DxT matrix
#Perform Tuning
#size_tune = 2
DocumentC_tsne = TSNE(n_components=2, verbose=1, random_state=0, perplexity=40,
                      angle=.99, n_iter =500,
                      init='pca').fit_transform(DxT_mat)

#Return the topic number that is the largest for each document
_lda_keys = []
for i in range(DxT_mat.shape[0]):
  _lda_keys +=  DxT_mat[i].argmax(),
_lda_keys = np.asarray(_lda_keys, dtype=None, order=None)

plot_embedding(DocumentC_tsne, _lda_keys, rescale=False)
plt.show()

#Perform on DxW matrix
DocumentC_tsne_DxW = TSNE(n_components=2, verbose=1, random_state=0, perplexity=40,
                      angle=.99, n_iter =500,
                      init='pca').fit_transform(df_tfidfC)




# New---------------------------

from utils import LowEmb, tsne, plot_embedding
from sklearn.manifold import TSNE

_lda_keys_lda7 = []

for i in range(DxT_mat_lda7.shape[0]):

  _lda_keys_lda7 +=  DxT_mat_lda7[i].argmax(),

_lda_keys_lda7 = np.asarray(_lda_keys_lda7, dtype=None, order=None)

 

_lda_keys_lda16 = []

for i in range(DxT_mat_lda16.shape[0]):

  _lda_keys_lda16 +=  DxT_mat_lda16[i].argmax(),

_lda_keys_lda16 = np.asarray(_lda_keys_lda16, dtype=None, order=None)

 

_lda_keys_lsi = []

for i in range(DxT_mat_lsi.shape[0]):

  _lda_keys_lsi +=  DxT_mat_lsi[i].argmax(),

_lda_keys_lsi = np.asarray(_lda_keys_lsi, dtype=None, order=None)

 

"""

t-SNE Dimensionality Reduction

"""

DocumentC_tsne_lda7 = TSNE(n_components=2, verbose=1, random_state=0, perplexity= 100,

                      angle=.99, n_iter =500,

                      init='pca').fit_transform(DxT_mat_lda7)

plot_embedding(DocumentC_tsne_lda7, _lda_keys_lda7, rescale=False)

plt.show()

 

DocumentC_tsne_lda16 = TSNE(n_components=2, verbose=1, random_state=0, perplexity= 60,

                      angle=.99, n_iter =500,

                      init='pca').fit_transform(DxT_mat_lda16)

plot_embedding(DocumentC_tsne_lda16, _lda_keys_lda16, rescale=False)

plt.show()

# t-SNE with tf-idf matrix ---------------------------- 
temp = TSNE(n_components=2, verbose=1, random_state=0, perplexity=100, angle=.99,
            n_iter=500, init='pca').fit_transform(df_tfidfC)

plot_embedding(DocumentC_tsne_lda7, _lda_keys_lda7, rescale=False)
# ----------------------------------------------------- 

#First perform on DxT matrix

#Perform Tuning

size_tune = 2

perp_list = [40,60,80,100]

for perp in perp_list:

    print(perp)

    Documentc_tsne_lda7 = TSNE(n_components=2, verbose=1, random_state=0, perplexity= perp,

                      angle=.99, n_iter =500,

                      init='pca').fit_transform(DxT_mat_lda7)

    plot_embedding(DocumentC_tsne_lda7, _lda_keys_lda7, rescale=False)

    plt.show()

#tuning for T=16
size_tune = 2

perp_list = [40,60,80,100]

for perp in perp_list:

    print(perp)

    DocumentC_tsne_lda16 = TSNE(n_components=2, verbose=1, random_state=0, perplexity= perp,

                      angle=.99, n_iter =500,

                      init='pca').fit_transform(DxT_mat_lda16)

    plot_embedding(DocumentC_tsne_lda16, _lda_keys_lda16, rescale=False)

    plt.show()

"""
Bokeh Visualization
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

import numpy as np
import bokeh.plotting as bp
from bokeh.io import output_notebook
from bokeh.plotting import figure, show, save
from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider
from bokeh.layouts import column

n_top_words = 5 # number of keywords we show

# 20 colors for 20 topics
colormap = np.array([
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
])

word_list = []
for k, words in dictionaryC.iteritems():
    word_list.append(words)

topic_summaries_lda7 = []
for i, topic_dist in enumerate(TxW_mat_lda7):
  TxW_mat_lda7 = np.array(word_list)[np.argsort(topic_dist)][:-(n_top_words + 1):-1] # get!
  topic_summaries_lda7.append(' '.join(TxW_mat_lda7)) # append!

topic_summaries_lda12 = []
for i, topic_dist in enumerate(TxW_mat_lda12):
  TxW_mat_lda12 = np.array(word_list)[np.argsort(topic_dist)][:-(n_top_words + 1):-1] # get!
  topic_summaries_lda12.append(' '.join(TxW_mat_lda12)) # append!

topic_summaries_lsi = []
for i, topic_dist in enumerate(TxW_mat_lsi):
  TxW_mat_lsi = np.array(word_list)[np.argsort(topic_dist)][:-(n_top_words + 1):-1] # get!
  topic_summaries_lsi.append(' '.join(TxW_mat_lsi)) # append!





title = '2010s Parliamentary Debates Visualization - LDA (7 Topics)'


num_example = len(DxT_mat_lda7)
list_lda_keys_lda7 = np.ndarray.tolist(_lda_keys_lda7)

plot_lda = bp.figure(plot_width=1200, plot_height=1000,
                     title=title,
                     tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                     x_axis_type=None, y_axis_type=None, min_border=1)

#Change the content to the topic title later
heading = []
for i in range(13206):
    with open("/Users/xiaoyusun/Downloads/TTIC 31220/Project/data2/C"+str(i+1)+".txt", 'r') as file:
        heading.append(file.readline().strip())

source = ColumnDataSource(data = pd.DataFrame(x=DocumentC_tsne_lda7[:, 0], y=DocumentC_tsne_lda7[:, 1],
                                      col=colormap[list_lda_keys_lda7][:num_example]),
    content = heading[:num_example], topic_key = list_lda_keys_lda7[:num_example])
plot_lda.scatter(x=DocumentC_tsne_lda7[:, 0], y=DocumentC_tsne_lda7[:, 1],
                 color=colormap[list_lda_keys_lda7][:num_example],
                 source=bp.ColumnDataSource({
                  "content": heading[:num_example], #to be changed
                   "topic_key": list_lda_keys_lda7[:num_example]
                   }))

"""
# randomly choose a news (within a topic) coordinate as the crucial words coordinate
topic_coord = np.empty((DxT_mat_lda12.shape[1], 2)) * np.nan
for topic_num in _lda_keys_lda12:
  if not np.isnan(topic_coord).any():
    break
  topic_coord[topic_num] = DocumentB_tsne_lda12[list_lda_keys_lda12.index(topic_num)]
# plot crucial words
for i in range(DxT_mat_lda12.shape[1]):
  plot_lda.text(topic_coord[i, 0], topic_coord[i, 1], [topic_summaries_lda12[i]])
"""

# hover tools
hover = plot_lda.select(dict(type=HoverTool))
hover.tooltips = {"content": "@content - topic: @topic_key"}

# save the plot
save(plot_lda, '{}.html'.format(title))



""""###########################################################
Document Clustering & Comparison
############################################################"""

# Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
hierarchicalCluster = AgglomerativeClustering(affinity='euclidean', 
                                              linkage='ward', n_clusters=10)
hierarchicalCluster.fit()




# Spectral Clustering
from sklearn.cluster import SpectralClustering
spectralCluster = SpectralClustering(n_clusters=10, assign_labels="discretize",
                                     gamma=1.0, random_state=0)
spectralCluster.fit()




dic_topic = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J"}












