import spacy
from scipy import spatial
import sys
import nltk
from nltk.stem import WordNetLemmatizer 
import re, string
# import preprocessor as p
# from preprocessor.api import clean, tokenize, parse
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import pandas as pd
import warnings
from langdetect import detect
from textblob import TextBlob


# print(len(sys.argv))
# sys.exit(0)

START = 0
LIMIT = 0


if len(sys.argv) != 3:
    print("You must pass the start and limit variable")
    sys.exit(0)   

# if sys.argv[1] is not int:
#     print("START must be an int")
#     sys.exit(0)

# if sys.argv[2] is not int:
#     print("LIMIT must be an int")
#     sys.exit(0)

START = int(sys.argv[1])
LIMIT = int(sys.argv[2])


fname = str(START) + "-" + str(LIMIT) + ".csv"


warnings.filterwarnings('ignore', '.*')

nlp = spacy.load("en_core_web_lg")
# nlp = spacy.load("en_vectors_web_lg")


translator = str.maketrans('', '', string.punctuation) 
stop_words = set(stopwords.words("english")) 

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    # d = df.iloc[i]
    blob = TextBlob(sentence)

    result = round(blob.sentences[0].sentiment.polarity, 3)
    # 0 - Neutral
    # 1 - Positive 
    # 2 - Very Positive
    # 3 - Negative
    # 4 - Vey Negative

    SENTIMENT = None
    
    #VERY NEGATIVE
    if(result > -0.5):
        SENTIMENT = 4

    # NEGATIVE
    if((result < 0) & (result <= -0.5)):
        SENTIMENT= 3

    # NEUTRAL
    if(result == 0):
        SENTIMENT = 0

   
    # POSITIVE
    if((result > 0) & (result <= 0.5)):
        SENTIMENT= 1

     #VERY POSITIVE
    if(result > 0.5):
        SENTIMENT = 2

    return SENTIMENT
  

def dataPreprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text) 
    text =  text.translate(translator) 
    text = " ".join(text.split()) 

    text = re.sub("[^a-zA-Z]+", " ", text)
    text = re.sub(r'http\S+', '', text)


    word_tokens = word_tokenize(text) 
    text = [word for word in word_tokens if word not in stop_words] 
    text = [lemmatizer.lemmatize(word) for word in text] 

    # text = [word for word in text if word not in nlp.vocab[word].vector] 
    # print()
    return text




categories = [["health", "medicine"], ["business", "finance"], ["education", "school"], ["travel", "tourism"], ["science", "technology"]]
# print(nlp('Religion').similarity(nlp('Beliefs')))

# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

df = pd.read_csv('data/cleaned/2020-04-19 Coronavirus Tweets.CSV')
df = df.iloc[START:]
# df = df.head(20000)

# print(df.head(5))
# df = df.head(5)

proccesed_data = []

for index_df in range(len(df)):


    inner_score_array = [0] * 5

    for i, new_cat in enumerate(categories):       
        clean_data  = dataPreprocess(df.iloc[index_df]['text'])
        # print(clean_data)

        for index, text in enumerate(clean_data):
                           
            cat1 = new_cat[0]
            cat2 = new_cat[0]
            
            similarity1 = nlp(text).similarity(nlp(new_cat[0]))
            similarity2 = nlp(text).similarity(nlp(new_cat[1]))

            if similarity1 > 0.5:
                inner_score_array[i] = 1
                break;


            if similarity2 > 0.5:
                inner_score_array[i] = 1
                break;
         

   
    if inner_score_array != [0, 0, 0, 0, 0, 0, 0] :
        inner_score_array.append(sentiment_analyzer_scores(' '.join(clean_data)))
        inner_score_array.append(' '.join(clean_data))
        # print(inner_score_array)
        proccesed_data.append(inner_score_array)
        print( str(index_df),  " >> ")


    if len(proccesed_data) == LIMIT :
        break;


df_output = pd.DataFrame (proccesed_data,columns=['health/medicine','business/finance','education/school','travel/tourism','science/technology','sentiment','text'])
df_output.to_csv('data/processed/' + fname)


print("Done >>>>>>>>>>>>>>>>>>>>>>.")