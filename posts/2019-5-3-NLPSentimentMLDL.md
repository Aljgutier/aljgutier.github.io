---
title: "Sentiment Classification Deep Learning and Machine Learning"
date: "2019-05-3 18:08:41 +0200"
author: Al Gutierrez
description: Sentiment classificaiton with Deep Learning, Jupyter notebook, Support Vector Machine, Fastai, PyTorch, State-of-the-Art performance, Universal Language Model
...

<span style="display:block; color:blue; margin-top:-90px;"> </span>
[about me](../about.md)  &nbsp;   &nbsp;  &nbsp;  &nbsp;   &nbsp;   &nbsp;  &nbsp;  &nbsp; [home](../index.md)
<figure>
 <img src="/images/NLP_Sentiment_MLDL/CinemaImage.png"  width="635">
</figure>


<h1 style="color:	#115BDC; text-align:center;">State of the Art Sentiment Classification with ML and DL models </h1>

May 3, 2019
 
We compare two sentiment classifiers, one based on a standard machine learning (ML) architecture built with Python's NLTK and Sklearn libraries and the other a deep learning (DL) model based on the [ULMFiT architecture](https://arxiv.org/abs/1801.06146). This ULM Sentiment Classifier builds on the Fastai library, a library which in turn utilizes [PyTorch](https://pytorch.org/). The objective is not to say one classifier is better than the other, but to understand state-of-the-art classification performance and the critical differences between the two classifiers. The goal is to demonstrate how to achieve world-class performance (deep learning or machine learning). This exercise is useful to applied data scientists interested in an easily accessible reference implementation with established benchmark performance.

### Dataset: IMDb, Large Movie Reviews
We use the [IMDb Large Movie Reviews](http://ai.stanford.edu/~amaas/data/sentiment/) dataset. The dataset has 3 classes positive, negative and unsupervised (sentiment unknown). There are 75k training reviews (12.5k positive sentiment, 12.5k negative sentiment, 50k unlabeled) and 25k validation reviews(12.5k positive, 12.5k). Refer to the README file in the IMDb corpus for further information about the dataset. 


### Additional Use Cases

Sentiment classification is a well-known text, NLP use case. However, the methods for NLP Sentiment classification are naturally adapted for new valuable use cases. Some use cases that follow from sentiment classification are:

* Legal document discovery, predicting discoverable documents from previously labeled documents.
* Fintech NLP sentiment and forecasting, for example, identify headlines or news documents that positively or negatively affect equities.
* Improved sales forecasts based on timely news sources
* Predicting actions based on customer service queries or support logs


<h2 style="color:	#115BDC;">ML Sentiment Classification </h2>

<figure>
 <img  alt="Machine Learning Sentiment Classificaion Pipeline" title="ML Sentiment Classification Pipeline" src="/images/NLP_Sentiment_MLDL/NLP_Sentiment_ML.png" width="635">
 <figcaption><center>Figure 1. NLP Sentiment Classification Pipeline</center></figcaption>
 </figure>


The ML sentiment classifier is illustrated in Figure 1, beginning with pre-processing, then Tokenization & Vectorization, followed by sentiment classification. The architecture references the following blog posts [Sentiment Analysis with Python, Part I](https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184) and [Sentiment Analysis with Python, Part II](https://towardsdatascience.com/sentiment-analysis-with-python-part-2-4f71e7bde59a). 

We begin by importing the the necessary python packages.

```
import html
from path import Path
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
```

We load the data, shuffle, and keep labeled documents.

```python
# Step 1: Load and Pre-process: Shuffle, Keep Labeled Data

PATH=Path('./data/aclImdb')
CLASSES = ['neg', 'pos', 'unsup']
def get_texts(path):
    texts,labels = [],[]
    for idx,label in enumerate(CLASSES):
        for fname in (path/label).glob('*.*'):
            texts.append(fname.open('r', encoding='utf-8').read())
            labels.append(idx)
    return np.array(texts),np.array(labels)


train_texts,train_labels = get_texts(PATH/'train') 
val_texts,val_labels = get_texts(PATH/'test')       


# Shuffle
np.random.seed(42)
train_idx = np.random.permutation(len(train_texts))
val_idx = np.random.permutation(len(val_texts))
train_texts = train_texts[train_idx]
val_texts = val_texts[val_idx]
train_labels = train_labels[train_idx]
val_labels = val_labels[val_idx]

# Keep Labeled Data
idx=np.where(train_labels != 2 )[0]
train_texts = train_texts[idx]
train_labels = train_labels[idx]


```

Next, punctuation and HTML fields are removed. The pre-processing results in two data frames `train_clean`, and `val_clean`. 

```python
# htmlfix function
re1 = re.compile(r'  +')
def htmlfix(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

punctuationfix = re.compile("[.;:!\'?,\"()\[\]]")

# Remove punctuation
# Remove html 
def preprocess_reviews(reviews):
    reviews = [punctuationfix.sub("", line.lower()) for line in reviews]
    reviews= [ htmlfix(line) for line in reviews]
    return reviews

# Clean
#  run the htmlfix and pre-process functions
train_clean = preprocess_reviews(train_texts) # train ... extract training and validation from this
val_clean = preprocess_reviews(val_texts)  # test holdout set

```

### Step 2: Tokenize and Vectorize

Following pre-processing the NLTK CountVectorizer, removes stop words, generates Ngrams of length 1, 2, 3, and vectorizes (numeric tokens).

```python
# Step 2:  Tokenize and Vectorize
print("CountvVectorizor ... ")
stop_words = ['in', 'of', 'at', 'a', 'the']
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
ngram_vectorizer.fit(train_clean)
# change the two inputs below to be reviews_train_clean_lem ... if you want to use lemmatized text
X = ngram_vectorizer.transform(train_clean)
X_val = ngram_vectorizer.transform(val_clean)
print("vecctorized:", X.shape)
```


	   CPU times: user 14.9 s, sys: 135 ms, total: 15 s
	   Wall time: 15 s
	   CPU times: user 14.7 s, sys: 104 ms, total: 14.8 s
	   Wall time: 14.8 s
	   vecctorized: (25000, 5443695)


### Step 3: Sentiment Classification

Following vectorization we train an SVM model (Support Vector Machine, linear kernel) on a MacBook Pro, 2.6 GHz Intel Core i7, with 32 G Ram.  The tokenization takes 14.1 s CPU time, and classification with the SKlearn linear SVM takes 6.1 s CPU time. 

```python
# Step 2:  Tokenize and Vectorize
stop_words = ['in', 'of', 'at', 'a', 'the']
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
ngram_vectorizer.fit(train_clean)
%time X = ngram_vectorizer.transform(train_clean)
%time X_val = ngram_vectorizer.transform(val_clean)
print("vecctorized:", X.shape)
```

          CPU times: user 7.91 s, sys: 210 ms, total: 8.12 s
          Wall time: 7.11 s
          Final Accuracy: 0.90024

```python
# Step 3 Sentiment Classification
msvc = LinearSVC(C=0.01)
msvc.fit(X, train_labels)
print ("Accuracy: %s"  % accuracy_score(val_labels, msvc.predict(X_val)))
```

	   Accuracy: 0.90024

The model achieves 90% accuracy on the test set. 

<h2 style="color:	#115BDC;"> ULM  Sentiment Classifier </h2>

<figure>
 <img alt="Deep Learning Sentiment Classificaion with Language Model" title="Deep Learning Sentiment Classificaion with Language Model" src="/images/NLP_Sentiment_MLDL/ULM_Sentiment.png" width="635">
 <figcaption><center>Figure 2. Universal Language Model Sentiment Classifier</center></figcaption>
 </figure>


The Deep-learning classifier is based on the (ULMFIT) [Universal Language Model with Fine Tuning](https://arxiv.org/abs/1801.06146) architecture and is illustrated in the figure above. ULMFiT comes from the Fastai initiative led by Jeremy Howard at the University of San Francisco, [State of the Art Text Classification with Universal Language Models](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html).  The key idea is Universal Language Model trained on a general corpus and then fine-tuned for a target task. The architecture consists of an embeddings layer and a Recursive Neural Network with LSTM cells. In this case, the target task is sentiment classification on the IMDb dataset. The sentiment classifier consists of two parts, a deep-learning language model, and an Artificial Neural Network (ANN) sentiment classifier. Herein we call this the ULM Sentiment Classifier.

This architecture and code overview is presented in [Fastai, DL2 lecture 10](https://forums.fast.ai/t/part-2-lesson-10-wiki/14364). In this post, the discussion focuses on the sentiment classification part. A pre-trained language model, trained as a separate exercise, is loaded by the ULM Sentiment Classifier. The first classifier layer consists of  ReLU activations followed by softmax activation layer that outputs a probability distribution over the target classes ("positive" and "negative" sentiment). The final output of the classifier corresponds to the largest probability from the softmax layer.

### Deep Learning Language Model

A brief description of the language model is useful in order to understand it within the context of the ULM Sentiment Classifier. In summary, a language model receives at its input a sequence of words. In this case, the sequence of words is a movie review and for each successive input word it attempts to predict the next word. Following the training of the ULM, the last layer is discarded, and replaced with the the sentiment classifier. A subsequent post will discuss the language model. In the meantime, a notebook for training the ULM is available on Github [ULM Notebook](https://github.com/Aljgutier/aljpspacedl2/blob/master/b-ULM-Sentiment.ipynb)


### Notebook Setup

Below, is the python code for defining and training the ULM Sentiment Classifier. A corrresponding Jupyter notebook is available on Github, [ULM Sentiment Classifier Notebook](https://github.com/Aljgutier/aljpspacedl2/blob/master/b-ULM-Sentiment_Classifier.ipynb).  We start by importing the Fastai library and setting high-level variables. We are running Fastai 0.7, see installations instructions here [Fastai 0.7](https://forums.fast.ai/t/fastai-v0-7-install-issues-thread/24652). In addition to importing Fastai, there are several helper functions located in "./code/sentiment\_imdb_helpers.py". For convenience, the helper functions are listed in the appendix, at the end of this post

```python
from fastai.text import *
import html
print(torch.__version__)
print(np.__version__)
# torch version should be pre 1.0 for compatibility with Fastai 0.7
# np version should be 1.15 for compatibility with Fastai 0.7

%run -i ./code/sentiment_imdb_helpers.py

PATH=Path('data/aclImdb/')
# in NLP you will see LM (Language Model) path by convention
LM_PATH=Path('data/imdb_lm/')
LM_PATH.mkdir(exist_ok=True)
# Clas Path and Col Names
CLAS_PATH=Path('data/imdb_clas/')
CLAS_PATH.mkdir(exist_ok=True)
chunksize=24000
col_names = ['labels','text']

```

### Load, pre-process, tokenize

The process starts with data preprocessing (HTML removed, and documents vectorized). This is a straightforward operation consisting of parsing with regexp then Spacy tokenizer. The processing is encapsulated in the `get_all()` helper functions (see Appendix, below). The Fastai library enhances the Spacy processing  ("Tokenizer") for multiprocessing, which significantly speeds up the processing. Any movie review document requires processing by `get_all()` prior to training of the ULM Sentiment Classifier or prior to Prediction. We also set the chunksize to 24,000 and pass it to Pandas to process a chunk of reviews at a time. This is especially necessary when training the LM (language model). 

Next, the vocabulary is loaded, where `itos` is a dictionary mapping of integer (token) to string token for each word in the vocabulary. This mapping is created as part of the language modeling training process. Next, `stoi`, the reverse mapping is generated. Text for each document is extracted from the data frame into the Numpy array `trn_class` and `trn_val` with the use of comprehensions. Training and validation labels are contained in `trn_labels` and `val_labels`. 

```python
 Code: DL Sentiment Classification Code
 
# Load Data
df_trn = pd.read_csv(CLAS_PATH/'train.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv(CLAS_PATH/'test.csv', header=None, chunksize=chunksize)

# Tokenize
tok_trn, trn_labels = get_all(df_trn, 1)
tok_val, val_labels = get_all(df_val, 1)

# Load Vocabulary
itos = pickle.load((LM_PATH/'tmp'/'itos.pkl').open('rb'))
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})

trn_clas = np.array([[stoi[o] for o in p] for p in tok_trn])
val_clas = np.array([[stoi[o] for o in p] for p in tok_val])

trn_labels = np.squeeze(trn_labels)
val_labels = np.squeeze(val_labels)

```

### Define the ULM Sentiment Classifier

As previously discussed, the ULM Sentiment Classifier consists of two parts, the pre-trained ULM ("backbone"), plus classifier ("custom head"). This is similar to transfer learning, for example as with a computer vision model, a pre-trained Deep Learning model is loaded with pre-trained weights followed by the addition of a task-specific output stage. The entire model (backbone + custom head) is then tuned for the specific task, sentiment classification.  We set the dimensions of backbone, the same as the pre-trained ULM model including embedding size of 400, 3 hidden layers (`nl` = 3), with 1150 activations each (`nh` = 1150). The bptt (backpropagate through time parameter) is set to 70. After setting the ULM Sentiment Classifier parameters, a data loader is created, `md`, where the dataset is passed to the data loader constructor, to generate one batch at a time.

The classifier, custom head, consists of two layers. The first layer of the classifier contains emb\_sz x 3 (1200) ReLU activations and the second layer softmax activation. The reason for the 3 x emb_sz activations is to receive 3 sets of activations from the ULM, corresponding to concatenated pooling. These 3 sets of activations correspond to the last hidden state of the ULM, H, `maxpool(H)`, and `meanpool(H)`, where maxpool and meanpool operate on as large history as available in the GPU memory. The optimization function includes Adam Optimization, with gradient clipping of 25 (to prevent divergence). The regularization function `reg_fn` helps to avoid overfitting. The `max_seq` is an important parameter that defines the maximum sequence handled by the GPU. The GPU memory needs to accommodate this sequence length.

A fastai `learner` object combines our data model loader (`md`) and RNN classifier (`m`) for which we can call `learner.fit()`.

```python
bptt,em_sz,nh,nl = 70,400,1150,3
vs = len(itos)
bs = 48 

md = make_ModelDataLoader(trn_clas, trn_labels, val_clas, val_labels, bs)
print(bs, bptt)

dps = np.array([0.4,0.5,0.05,0.3,0.4])*0.5
# change 20*70 to 10*70 ... running out of memory with 20 * 70 ... see notes/comments below

c=int(trn_labels.max())+1
max_seq=10*bptt
m = get_rnn_classifer(bptt, max_seq, c, vs, emb_sz=em_sz, n_hid=nh, 
                      n_layers=nl, pad_token=1,
                      layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
                      dropouti=dps[0], wdrop=dps[1],        
                      dropoute=dps[2], dropouth=dps[3])
opt_fn = partial(optim.Adam, betas=(0.7, 0.99))

learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=partial(optim.Adam, betas=(0.7, 0.99)))
learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learn.clip=25.
learn.metrics = [accuracy]
```
### Train

***Learn - last layer***  

We are now ready to start learning. The ULMFiT model employs a gradual unfreezing approach, wherein first, the last layer ("classifier") weights are unfrozen, and the corresponding weights are adjusted. After one training epoch, we achieve a 92.88 % accuracy. This result is already better than ML example above. The learning rates are specified in a Numpy array, where each learning rate corresponds to a specific network layer from the first layer to the last layer, a technique called "discriminative fine-tuning."


```python
wd = 0
lrs=np.array([1e-4,1e-4,1e-4,1e-3,1e-2])

# load language model
learn.load_encoder('lm1_enc')  # this model is saved under PATH/models/lm1_enc

learn.freeze_to(-1)
learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))
# bs = 48, bptt = 70
```

           epoch      trn_loss   val_loss   accuracy                      
             0          0.273541   0.182889   0.92888 

***Learn - unfreeze one more layer***

We then unfreeze the next layer and train for an additional epoch, achieving a 93.6% accuracy.

```python
learn.freeze_to(-2)
learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))
# bs = 48, bptt = 70
``` 

           epoch      trn_loss   val_loss   accuracy                      
             0           0.230917   0.165796   0.93692  

***Learn - unfreeze all layers***

Unfreezing the entire classifier, allows for the adjustment of all weights, from the input to the output layer, and thereby acheiving state of the art accuracy of 94.8%.

Training of the Language Model required on the order of 20 hours on a [Paperspace](https://www.paperspace.com) P4000 virtual desktop resource  consisting of an 8 Gbyte NVIDIA, P4000, GPU, and 30 GB, Intel Xeon E5-2623 v4 CPU. After loading the pre-trained language model, the ULM Sentiment Classifier required approximately 4 hours of additional training time. 

```python
learn.unfreeze()
learn.fit(lrs, 1, wds=wd, cycle_len=14, use_clr=(32,10))
# bs = 48, bptt = 70
```


         epoch      trn_loss   val_loss   accuracy                                            
           13        0.165997   0.146615   0.947905


<h2 style="color:	#115BDC;">Summary of Results: DL vs ML Sentiment Classification </h2>

In summary, we see a significant improvement in predictive performance provided by DL over ML sentiment classification. The salient characteristics of each classifier are summarized in the table below. Each of the classification models achieved state-of-the-art performance on the respective domain, ML with NLTK and Sklearn, or Deep-Learning. 

Beginning with the ML Sentiment classifier, it cleans and processes the data followed by removing stop words, removing punctuation, and creating Ngrams (1, 2, 3 words). The resulting vectorized tokens are then used to train a linear SVM classifier. In contrast, the ULM Sentiment Classifier develops an understanding of the language so that stop words are not removed (or lemmatized). Punctuation characteristics are captured with tokens, such as "BOS" (Beginning of Sentence).

Relative to the ML classifier, the ULM Sentiment classifier employs several novel methods, as presented in 
[ULMFit](https://arxiv.org/pdf/1801.06146.pdf).  

 * Language Model. A pre-trained language model, trained on a large general domain corpus, Wiki, then fine-tuned for the target task.  
 * Discrimitive fine-tuning. Different layers hold different types of information. Instead of tuning all layers at the same rate, discriminative learning is applied, that is, different layers are tuned at different rates.
 * Gradual Unfreezing. Tuning all layers at once risks catastrophic forgetting of the pre-trained information. Therefore, first, the last layer is unfrozen and trained, then the next to last layer, and so on. 
 * Slanted Triangular Learning. This method allows the model to quickly adapt to quickly converge to a suitable region of the parameter space.
 * Concatenated Pooling. Since the signal in text classification is often contained in a few words, which may occur anywhere in the document, the method of concatenated pooling is employed. 

The ULM Sentiment classifier achieves 94.8% accuracy, which outperforms the classification accuracy of all models published before the ULMFit model (94.1%). For instance A recent paper from Bradbury et al, [Learned in translation: contextualized word vectors](https://arxiv.org/pdf/1708.00107.pdf), has a handy summary of the latest academic research in solving this IMDB sentiment analysis problem, where many of the latest algorithms shown are tuned for this specific problem.

The State of the art ULMFit model adds one more technique for achieving even better accuracy.  It trains two models:  one with a language model trained in the forward direction and another model trained by reversing the order of the text. Then, the final prediction is based on the average prediction of each model. This addition leads to a 95.4% accuracy.


<table>
 <caption>Table 1. ML and DL Sentiment classifier Comparison Summary</caption>
    <tr>
     <td width="14%"  style="text-align:center;vertical-align:center;" > <strong>Characteristic</strong></td>
        <td width="40%" style="text-align:center;vertical-align:center;" ><strong>ML Sentimeent Classifier</strong></td>
        <td width="46%" style="text-align:center;vertical-align:center;" ><strong>ULM Sentiment Classifier</strong></td>
    </tr>
    <tr>
        <td>Architecture</td>
        <td style="text-align:left;vertical-align:top;" >
           - Data pre-processing - filter html and punctuation. <br>
           - Tokenization & Vectorization: Ngrams (1,2,3), stop-word removal. <br>
           - Classifier: SVM sentiment classification </td>
        <td style="text-align:left;vertical-align:top;">
           - Data pre-processing: filter html <br>
           - Tokenization & Vectorization: punctuation markers (e.g., "BOS" and Capitilization) <br>
           - Classifier: ULM Sentiment Classifier including Deep-Learning Language Model backbone + Artificial Neural Network Sentiment Classifier custom head 
        </td>
   </tr>
   <tr>
        <td >Training</td>
        <td style="text-align:left;vertical-align:top;" > - CPU (Macbook Pro, 2.6 GHz Intel Core i7, with 32 G Ram <br>
            ~ minute
        </td>
        <td>- Paperspace P4000 virtual desktop: NVIDIA P4000, 8 GB GPU, 1791 CUDA Cores <br>
           ~24 hours: ~20 hours ULM, ~4 hours ANN Classifier
            
        </td>
    </tr>
   <tr>
        <td>Error Rate</td>
        <td>10%</td>
        <td>5.2%</td>
    </tr>
       <tr>
        <td>Differentiators</td>
        <td style="text-align:left;vertical-align:top;">
        - Training efficiency with 90% accuracy</td>
        <td>- 94.8% accuracy<br>
        - Deep Language Model </br>
        - Concatenated Pooling </br>
        - Gradual unfreezing </br>
        - Discriminitive fine-tuning <br>
        - Slanted triangular learning <br>
        </td>
    </tr>
</table>

<h1 style="color:	#115BDC;"> Appendix - Helper Functions </h1>


#### get_all(), pre-processing, cleaning, and tokenization for DL Classifier
```python
re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

    
def get_texts(df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)  # BOS beginning of text for a doc
    for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)  # multiple fields
    texts = list(texts.apply(fixup).values)
      # tokenize with process all multiprocessor ... tokenize slow, but speed up with multiple cores
      # SpaCy gret, but slow and with multi processor its much better
      # number of sublists is number of cores on your computer ... each part of the list will be tokenized on different core
      #   on Jeremy's machine 1.5 hours without mulitprocessing, a couple of minutes with multiprocessing
      #   e.g. we all have multicores on our laptops
    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)
    
def get_all(df, n_lbls):
    tok, labels = [], []
       # go through each chunck (each is a dataframe) and call get_texts
       #    get_texts will grab labels make them into ints and grb texts 
       #    before including the text get_text includes BOS function.
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels

```

#### make_ModelDataLoader()


```python
def make_ModelDataLoader(trn_clas, trn_labels, val_clas, val_labels, bs):
    min_lbl = trn_labels.min()
    trn_labels -= min_lbl
    val_labels -= min_lbl
    trn_ds = TextDataset(trn_clas, trn_labels)
    val_ds = TextDataset(val_clas, val_labels)
    trn_samp = SortishSampler(trn_clas, key=lambda x: len(trn_clas[x]), bs=bs//2)
    val_samp = SortSampler(val_clas, key=lambda x: len(val_clas[x]))
    trn_dl = DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
    val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
    md = ModelData(PATH, trn_dl, val_dl)
    return md

```

