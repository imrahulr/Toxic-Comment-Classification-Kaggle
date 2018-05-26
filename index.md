# Toxic Comment Classification Challenge

####Identify and classify toxic online comments


Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.

So, in this competition on Kaggle, the challenge was to build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate better than Perspective’s current models. A dataset of comments from Wikipedia’s talk page edits was provided. 


## Data Overview:

The dataset used was Wikipedia corpus dataset which was rated by human raters for toxicity. The corpus contains comments from discussions relating to user pages and articles dating from 2004-2015.

The comments are to be tagged in the following six categories - 
    +toxic
    +severe_toxic
    +obscene
    +threat
    +insult
    +identity_hate
    
###Train and Test Data

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

#####Train Data after basic preprocessing and cleaning

id 	comment_text 	toxic 	severe_toxic 	obscene 	threat 	insult 	identity_hate
0 	0000997932d777bf 	explanation why the edits made under my userna... 	0 	0 	0 	0 	0 	0
1 	000103f0d9cfb60f 	d aww ! he matches this background colour i am... 	0 	0 	0 	0 	0 	0
2 	000113f07ec002fd 	hey man i am really not trying to edit war it ... 	0 	0 	0 	0 	0 	0
3 	0001b41b1c6bb37e 	more i cannot make any real suggestions on im... 	0 	0 	0 	0 	0 	0
4 	0001d958c54c6e35 	you sir are my hero any chance you remember wh... 	0 	0 	0 	0 	0 	0

#####Test Data after basic preprocessing and cleaning

