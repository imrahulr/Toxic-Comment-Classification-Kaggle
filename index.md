# Toxic Comment Classification Challenge

#### Identify and classify toxic online comments



Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.

So, in this competition on Kaggle, the challenge was to build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate better than Perspective’s current models. A dataset of comments from Wikipedia’s talk page edits was provided. 

---

## Data Overview:

The dataset used was Wikipedia corpus dataset which was rated by human raters for toxicity. The corpus contains comments from discussions relating to user pages and articles dating from 2004-2015.

The comments are to be tagged in the following six categories - 
<ul>
    <li>toxic</li>
    <li>severe_toxic</li>
    <li>obscene</li>
    <li>threat</li>
    <li>insult</li>
    <li>identity_hate</li>
</ul>

### Train and Test Data

The training data contains a row per comment, with an id, the text of the comment, and 6 different labels that we have to predict.

```python
import pandas as pd
import numpy as np

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print('Train shape: ', train_df.shape)
print('Test shape: ', test_df.shape) 
```

Train shape:  (159571, 8)
Test shape:  (153164, 2)

### Train Data after basic preprocessing and cleaning

|   | id | comment_text | toxic | severe_toxic | obscene | threat | insult | identity_hate |
|:--:|:---------------:|--------------|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0000997932d777bf |	explanation why the edits made under my userna... |	0 |	0 |	0 |	0 |	0 |	0 |
| 1 | 000103f0d9cfb60f |	d aww ! he matches this background colour i am... |	0 |	0 |	0 |	0 |	0 |	0 |
| 2 | 000113f07ec002fd |	hey man i am really not trying to edit war it ... |	0 |	0 |	0 |	0 |	0 |	0 |
| 3 | 0001b41b1c6bb37e |	more i cannot make any real suggestions on im...  |	0 |	0 |	0 |	0 |	0 |	0 |
| 4 | 0001d958c54c6e35 |	you sir are my hero any chance you remember wh... |	0 |	0 |	0 |	0 |	0 |	0 |

### Test Data after basic preprocessing and cleaning

|  | id | comment_text |
|:--:|:--------:|-------|
| 0 | 00001cee341fdb12| 	yo bitch ja rule is more succesful then you wi... |
| 1 | 0000247867823ef7| 	= = from rfc = = the title is fine as it is imo |
| 2 | 00013b17ad220c46| 	= = sources = = zawe ashton on lapland |
| 3 | 00017563c3f7919a| 	if you have a look back at the source the inf... |
| 4 | 00017695ad8997eb| 	i do not anonymously edit articles at all |

### Cleaning Data

```python
def cleanData(text, stemming=False, lemmatize=False):    
    text = text.lower().split()
    text = " ".join(text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+\-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    if stemming:
        st = PorterStemmer()
        txt = " ".join([st.stem(w) for w in text.split()])
    if lemmatize:
        wordnet_lemmatizer = WordNetLemmatizer()
        txt = " ".join([wordnet_lemmatizer.lemmatize(w) for w in text.split()])
    return text
```

### Exploring Train Data

#### Number of Occurrences of each Output Class

<p align="center">
<img src="/img/noofoccurrences.png" alt="Number of Occurrences of each Class"/>
</p>

#### Correlation between Output Classes

<p align="center">
<img src="/img/corr.png" alt="Correlation between Output Classes"/>                                                                                                                                  </p>

#### Words frequently occurring in Toxic Comments

<p align="center">
<img src="/img/wordtoxic.png" alt="Words frequently occurring in Toxic Comments"/>
</p>
                                                                                                                                                      
#### Words frequently occurring in Severe Toxic Comments

<p align="center">
<img src="/img/wordstox.png" alt="Words frequently occurring in Severe Toxic Comments"/>
</p>

#### Words frequently occurring in Threat Comments

<p align="center">
<img src="/img/woedthreat.png" alt="Words frequently occurring in Threat Comments"/>
</p>

#### Words frequently occurring in Insult Comments

<p align="center">
<img src="/img/wordinsult.png" alt="Words frequently occurring in Insult Comments"/>
</p>

---

## Our solution


The final solution consists of ensemble of several machine learning models - 

<ul>
<li>Attention with Bidirectional LSTM</li>
<li>Bidirectional LSTM with Pre-Post Input Text</li>
<li>Bidirectional GRU with derived features</li>
<li>CNN based on DeepMoji Architecture</li>
<li>CNN + GRU</li>
<li>DeepMoji Architecture</li>
<li>Character Level Hierarchical Network</li>
<li>Ensemble of Logistic Regression and SVM</li>
<li>2D CNN</li>
<li>LightGBM</li>
</ul>

Each model was trained using 10 fold validation with proper hyperparameter tuning. We used LightGBM and simple weighted averaging for stacking these models.

### Embeddings Used

Various pre-trained embeddings were used to create diverse models -
<ul>
<li>GloVe</li>
<li>fastText</li>
<li>word2vec</li>
<li>Byte-Pair Encoded subword embeddings (BPE)</li>
</ul> 

---

## Results

<ul>
<li>The overall model got a ROC AUC score of 0.9874 on private LB.</li>
<li>Preprocessing was not much impactful and did not significantly improve the score of any model.</li>
<li>RNN models were significantly better than CNN models.</li>
<li>The best model was DeepMoji followed by CNN-GRU.</li>
<li>Adding attention layer to RNN models boosted their score.</li>
<li>Logistic regression and LightGBM models had much lower scores but provided diversity.</li>
</ul>

### Thank You!
