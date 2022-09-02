# Sentiment Analysis for Reviews of Chartered Bus

A chartered bus is a private vehicle rental used by private groups to take them, and only them, to their destination. With a charter bus, your group has the vehicle to themselves and gets to customize their itinerary as much as they’d like. Charter buses typically feature perks like a driver, undercarriage storage space, onboard bathrooms, free WiFi, and more.

![Untitled](img/Untitled.png)

The application of Chartered Bus is on Google Play Store and the reviews have been scrapped from there only. This project aims to build a sentimental analysis of customers’ experiences, being an active user of Chartered Bus I loved to choose this app. Chartered Bus is the most popular in Madhya Pradesh. While reading the dataset I obtained from Play Store, it is some mixed reactions from users. Although, the company promises to keep Reliability, Quality, Responsibility, Sincerity, Vitality, and Strong Business Ethics, let's see how the people of India look into this.

### **Let's dive into this amazing project and try to extract the real thoughts of the chartered users.**

***Problem Statement:* Make a python script that fetches the App reviews(500 latest ones) for any app that you like and does basic sentiment analysis on those reviews.**

First of all, we are importing all necessary Libraries.

```python
from google_play_scraper import app
import pandas as pd
import numpy as np
```

Fetching reviews data from Play Store of Chartered App.

Source : 

[Chartered Bus - Apps on Google Play](https://play.google.com/store/apps/details?id=com.bitla.mba.charteredbus)

```python
from google_play_scraper import Sort, reviews_all

chartered_review = reviews_all(
    'com.bitla.mba.charteredbus',
    sleep_milliseconds=0,
    lang='en',
    country='in',
    sort=Sort.NEWEST,
)
```

Changing the obtained data into a data frame and naming it as df, printing the first five rows.

```python
df=pd.DataFrame(chartered_review)
df.head(5)
```

![Untitled](img/Untitled%201.png)

```python
df.columns
```

```
Output

Index(['reviewId', 'userName', 'userImage', 'content', 'score',
       'thumbsUpCount', 'reviewCreatedVersion', 'at', 'replyContent',
       'repliedAt'],
      dtype='object')
```

 So the total number of columns is 10, and out of all the rating text is in the ‘**content’.**

The rest of the columns one can ignore for now because for doing sentiment analysis, we have to do nothing with that.

## Cleaning the Dataset

The Dataset should be needed to clean first. The model we are going to use is specified for English texts only, but in chartered bus reviews, we can see a lot of data in the Hindi language. So our very first work is to refine those, with an eye analysis it can be seen that Hindi reviews normally extend the length of 15-20 words, however, English reviews are limited to ten words only sometimes.

So one job to do is to filter only those rows from the dataset whose content column contains words of less than 10 lengths.

For that we have implemented a single line of code:

```python
df=(df[df['content'].apply(lambda x: len(x.split()) < 10)])
df
```

## Sentiment Analysis Model

```python
from transformers import pipeline
sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
```

This model ("SiEBERT", prefix for "Sentiment in English") is a fine-tuned checkpoint of **[RoBERTa-large](https://huggingface.co/roberta-large))**. It enables reliable binary sentiment analysis for various types of English-language text. For each instance, it predicts either positive (1) or negative (0) sentiment. The model was fine-tuned and evaluated on 15 data sets from diverse text sources to enhance generalization across different types of texts (reviews, tweets, etc.). Consequently, it outperforms models trained on only one type of text (e.g., movie reviews from the popular SST-2 benchmark) when used on new data.

Source: Huggingface

The model has been taken from the Huggingface website and it's called siebert/sentiment-roberta-large-english.

**How it works :** 

BERT takes as input a concatenation of two segments (sequences of tokens), x1, . . . , xN and y1, . . . , yM. Segments usually consist of more than one natural sentence. The two segments are presented as a single input sequence to BERT with special tokens delimiting them: [CLS], x1, . . . , xN , [SEP], y1, . . . , yM, [EOS]. M and N are constrained such that M + N < T, where T is a parameter that controls the maximum sequence length during training.

The model is first pre-trained on a large unlabeled text corpus and subsequently finetuned using end-task labeled data.

Performance: To evaluate the performance of our general-purpose sentiment analysis model, we set aside an evaluation set from each data set, which was not used for training. On average, our model outperforms a **[DistilBERT-based model](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)** (which is solely fine-tuned on the popular SST-2 data set) by more than 15 percentage points (78.1 vs. 93.2 percent, see table below). As a robustness check, we evaluate the model in a leave-one-out manner (training on 14 data sets, evaluating on the one left out), which decreases model performance by only about 3 percentage points on average and underscores its generalizability. Model performance is given as evaluation set accuracy in percent.

## ## Adding New Column Result

```python
df['result']=df['content'].apply(lambda text :sentiment_analysis(text) )
df
```

Output:

![Untitled](img/Untitled%202.png)

As we can see the last column , resul is actually a list in which there is a dictionary with key-value pairs of  label and score.

```python
[{'label': 'NEGATIVE', 'score': 0.999500274658}]
```

Our next job is to separate the label and score, so making two additional columns for this with the name Sentiment and Probability.

**Sentiment:** It defines the sentiment of the user, whether it is positive or negative**.**

**Probability:** It defines the probability of the result that is Positive or Negative is how True.

```python
df['sentiment']=df['result'].apply(lambda x : x[0]['label'])
df['probability']=df['result'].apply(lambda x : x[0]['score'] )
df
```

Small Snippet of that:

![Untitled](img/Untitled%203.png)

At last showing the result on pie chart.

```python
import matplotlib.pyplot as plt
a=plt.pie(df['sentiment'].value_counts(),labels=['Positive Review','Negative Review'],shadow=True,normalize=True)
```

Output:

![Untitled](img/Untitled%204.png)

Note: please refer to the jupyter notebook for code reference.
