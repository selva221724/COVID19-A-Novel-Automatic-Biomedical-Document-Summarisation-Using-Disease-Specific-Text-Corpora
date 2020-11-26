# import nltk
import yake


data='/mnt/dash/Alpha_Share/Automation_Team/Tamil/NLP_learning/rake-nltk/data/data.txt'
with open(data, encoding='utf-8') as f:
    text = f.read()


kw_extractor = yake.KeywordExtractor()


language = "en"
max_ngram_size = 3
deduplication_thresold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 20

custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords = custom_kw_extractor.extract_keywords(text)



