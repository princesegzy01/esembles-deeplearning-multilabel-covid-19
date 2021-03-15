import pandas as pd
import re, string
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords                   #Stopwords corpus
from nltk.stem import PorterStemmer 
from langdetect import detect
import preprocessor as p
import sys
import os
import glob

path = "data/raw/"
dirs = os.listdir(path)





# for fname in glob.glob(path):
#     print(fname)
stemmer = PorterStemmer() 

def preprocessing(document):
    
    # convert to lower case
    document = document.lower()

    # remove numbers
    document = re.sub(r'\d+', '', document) 

    
    
    # remove punctuation
    # translator = str.maketrans('', '', string.punctuation) 
    # document =  document.translate(translator) 

    # remove whitespace
    document = " ".join(document.split())

    document = re.sub(r"http\S+", "", document)

    word_tokens = word_tokenize(document) 

    # remove stop words
    stop_words = set(stopwords.words("english")) 
    word_tokens = word_tokenize(document) 
    filtered_text = [word for word in word_tokens if word not in stop_words] 

    # stemming
    # stems = [stemmer.stem(word) for word in filtered_text] 
    # document = ' '.join(stems)

    document = ' '.join(filtered_text)
    return document


for file in dirs:
    # print(file)

    # df = pd.read_csv("2020-04-19.CSV", sep=',')

    df = pd.read_csv("data/raw/" + file, sep=',')
    # df = df.head(1000)
    df2 = df[df['lang'] == "en"]
    # sample_data.to_csv("sample_data.csv")
    # print(len(df2))
    # print(len(df))

    df2 = df2.drop(['status_id','user_id','created_at','screen_name','source','reply_to_status_id','reply_to_user_id','reply_to_screen_name','is_quote','is_retweet','favourites_count','retweet_count','place_full_name','place_type', 'followers_count','friends_count', 'account_lang', 'account_created_at','country_code', 'verified','lang'], axis=1)
    print("Total before reset index " + str(len(df2)))


    df2 = df2.reset_index(drop=True)

    print("Total after reset index " + str(len(df2)))

    new_clean_list = []
    # for i, d in df2.iterrows():
    for i in range(0, len(df2)):

        d = df2.iloc[i]
        # print(d['text'])
        try:
            if detect(preprocessing(d['text'])) != "en":
                # df2.drop(df2.index[i])
                continue

            new_clean_list.append(d['text'])

        except:
            print("Error occour at " + str(i))

        
    print("Total after en" + str(len(new_clean_list)))

    newDfList = pd.DataFrame(new_clean_list) 
    newDfList.to_csv(file)

    print("done")
# print(len(l))