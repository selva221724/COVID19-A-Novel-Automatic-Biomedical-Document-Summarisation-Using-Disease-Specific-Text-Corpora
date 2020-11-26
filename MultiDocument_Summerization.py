#!/usr/bin/env python
# encoding: utf-8

import logging
import pytextrank
import spacy
import sys
import os
from pathlib import Path
from natsort import natsorted
######################################################################
## sample usage
######################################################################

# load a spaCy model, depending on language, scale, etc.

# nlp = spacy.load("en_ner_bionlp13cg_md")
nlp = spacy.load("en_core_web_sm")


text_path = '/mnt/dash/Alpha_Share/Automation_Team/Tamil/NLP_learning/pytextrank/data/1'
out_path = '/mnt/dash/Alpha_Share/Automation_Team/Tamil/NLP_learning/pytextrank/data/out'
human_summer = '/mnt/dash/Alpha_Share/Automation_Team/Tamil/NLP_learning/pytextrank/data/covid-19/abstract summarization.txt'

folders =[]
for i,j,k in os.walk(text_path):
    for fold in j:
        folders.append(os.path.join(i,fold))


docs =[]
for fold in folders:
    temp=[]
    for i,j,k in os.walk(fold):
        for doc in k:
            temp.append(os.path.join(i,doc))

    temp = natsorted(temp)
    docs.append(temp)

# with open('/mnt/dash/Alpha_Share/Automation_Team/Tamil/NLP_learning/pytextrank/pytextrank/ontalogy.txt', encoding='utf-8') as f:
#     key_terms = f.read()
#     key_terms = key_terms.split('\n')[:-1]
#
# key_dict ={}
# for i in key_terms:
#     key_dict.update({i:1})



def all_text(text_list):
    text_all = ''

    for i in text_list:
        # print(i)
        try:
            with open(i, encoding='utf-16') as f:
                text = f.read()
                text = text.replace('\n','')
                text_all+=text
        except:
            with open(i, encoding='utf-8') as f:
                text = f.read()
                text = text.replace('\n','')
                text_all+=text


    return text_all

def summerize(text, phrase, sentance) :
    doc = nlp(text)
    sum_text=''
    for sent in doc._.textrank.summary(limit_phrases=phrase, limit_sentences=sentance):
        sum_text += str(sent)
    return sum_text

def keywords(text,count):
    doc = nlp(text)
    sum_text = []
    for phrase in doc._.phrases[:count]:
        sum_text.append(str(phrase))
    return sum_text

# logging is optional: to debug, set the `logger` parameter
# when initializing the TextRank object

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("PyTR")

# add PyTextRank into the spaCy pipeline

tr = pytextrank.TextRank(logger=None)
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

for i in docs:
    text_extract = all_text(i)
    key_word = keywords(text_extract, 20)
    sumerize1 = summerize(text_extract,100,15)
    path_name = Path(str(Path(i[0]).parent)).name
    # text_file = open(out_path +'/'+ path_name+".txt", "w")
    # text_file.write(sumerize1)
    # text_file.close()


# text_file = open("/mnt/dash/Alpha_Share/Automation_Team/Tamil/NLP_learning/rake-nltk/data/data.txt", "w")
# text_file.write(text_extract)
# text_file.close()



# ------------------------------------------ Modified Approch -------------------------------
main_txt = ''
key_word_mod = []
for i in docs[0]:
    # print(i)
    b = all_text([i])
    a = summerize(b, 20, 2)
    key_word_mod.append(keywords(b, 20))
    main_txt+=a

sumerize = main_txt

# path_name = Path(str(Path(i[0]).parent)).name
# text_file = open(out_path +'/'+ path_name+".txt", "w")
# text_file.write(main_txt)
# text_file.close()

# ------------------------------------------- ROUGH ---------------------------------------------

with open(human_summer, encoding='utf-8') as f:
    human_summery = f.read()
    human_summery = human_summery.replace('\n', '')

import rouge

def prepare_results(p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

for aggregator in ['Avg', 'Best', 'Individual']:
    print('Evaluation with {}'.format(aggregator))
    apply_avg = aggregator == 'Avg'
    apply_best = aggregator == 'Best'

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                           max_n=4,
                           limit_length=True,
                           length_limit=100,
                           length_limit_type='words',
                           apply_avg=apply_avg,
                           apply_best=apply_best,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)


    hypothesis_1 = sumerize
    references_1 = human_summery
    all_hypothesis = [hypothesis_1]
    all_references = [references_1]

    scores = evaluator.get_scores(all_hypothesis, all_references)

    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
            for hypothesis_id, results_per_ref in enumerate(results):
                nb_references = len(results_per_ref['p'])
                for reference_id in range(nb_references):
                    print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                    print('\t' + prepare_results(results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
            print()
        else:
            print(prepare_results(results['p'], results['r'], results['f']))
    print()


# plotting ===============================================

import matplotlib.pyplot as plt

one_1 =[30.23,31.85,30.40]
one = [34.23,35.85,35.02]
two =[45.13,48.11,46.58]
three =[58.93,62.26,60.55]

p = [30.23,34.23,45.13,58.93]
r = [31.85,35.85,48.11,62.60]
f1 = [30.40,35.02,46.58,60.55]

x_axis = ['TextRank with Gensim','TextRank with Scapy','Individual-Document TextRank with Scapy','Modified TextRank with Scapy & YAKE']

fig, ax = plt.subplots()

ax.plot(x_axis, p, label="Precision",linestyle=':')
ax.plot(x_axis, r, label="Recall",linestyle='-')
ax.plot(x_axis, f1, label="F1-Measure",linestyle= '--')
ax.legend()
ax.set_title('Method Cross Comparison Chart',fontsize=14,fontweight='bold')
plt.xlabel('Algorithm Name',fontweight='bold')
plt.ylabel('Score',fontweight='bold')
plt.show()
