import re
import pandas as pd
from pdfminer.high_level import  extract_text  #need to pip install pyminer-six along with pyminer

text = extract_text("chapter-2.pdf")
text

# pattern = re.compile(r"[a-zA-z0-9]")
text = re.sub("[^a-zA-z0-9]"," ",text)
text = re.sub('[^?<!\w)\d+]', " ",text)
text = re.sub('[^-(?!\w)|(?<!\w)-]', " ",text)
text = " ".join(text.split())
text

from textwrap3 import wrap

import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_model = summary_model.to(device)

import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(10)

import nltk
nltk.download('punkt')
nltk.download('brown')
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize

def Capitalize (content):
  final=""
  for sent in sent_tokenize(content):
    sent = sent.capitalize()
    final = final +" "+sent
  return final

def summarizer(text,model,tokenizer):
  text = text.strip().replace("\n"," ")
  text = "summarize: "+text
  # print (text)
  max_len = 1000
  encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,
  truncation=True,return_tensors="pt").to(device)

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,attention_mask=attention_mask,early_stopping=True,
  num_beams=3,num_return_sequences=1,no_repeat_ngram_size=2,min_length = 75,max_length=300)
  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
  summary = dec[0]
  summary = Capitalize(summary)
  summary= summary.strip()

  return summary

summarized_text = summarizer(text,summary_model,summary_tokenizer)
summarized_text

from keybert import KeyBERT
kw_model = KeyBERT(model='all-mpnet-base-v2')

def keywords(paragraph):
   keywords = kw_model.extract_keywords(paragraph,keyphrase_ngram_range=(1, 3),stop_words='english', 
   highlight=False,top_n=15)
   keywords_list= list(dict(keywords).keys())
#    print(keywords_list)
   return keywords_list

word1 = keywords(text)
# word1.append(keywords(text))
word2 = keywords(summarized_text)
# word2.append(keywords(summarized_text))

question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_model = question_model.to(device)

def get_question(context,answer,model,tokenizer):
  text = "context: {} answer: {}".format(context,answer)
  encoding = tokenizer.encode_plus(text,max_length=1000, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=72)
  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
  Question = dec[0].replace("question:","")
  Question= Question.strip()
  return Question

# for wrp in wrap(summarized_text, 150):
#   print (wrp)
# print ("\n")

for answer in word2:
  ques = get_question(summarized_text,answer,question_model,question_tokenizer)
  print (ques)
  print (answer)


import gradio as gr

context = gr.inputs.Textbox(lines=10, placeholder="Enter paragraph/content here...")
output = gr.outputs.HTML(  label="Question and Answers")


def generate_question(context):
  summary_text = summarizer(context,summary_model,summary_tokenizer)
  for wrp in wrap(summary_text, 150):
    print (wrp)
  np =  keywords(context,summary_text)
  print ("\n\nNoun phrases",np)
  output=""
  for answer in np:
    ques = get_question(summary_text,answer,question_model,question_tokenizer)
    # output= output + ques + "\n" + "Ans: "+answer.capitalize() + "\n\n"
    output = output + "<b style='color:blue;'>" + ques + "</b>"
    # output = output + "<br>"
    output = output + "<b style='color:green;'>" + "Ans: " +answer.capitalize()+  "</b>"
    output = output + "<br>"

  summary ="Summary: "+ summary_text
  for answer in np:
    summary = summary.replace(answer,"<b>"+answer+"</b>")
    summary = summary.replace(answer.capitalize(),"<b>"+answer.capitalize()+"</b>")
  output = output + "<p>"+summary+"</p>"
  
  return output

iface = gr.Interface(
  fn=generate_question, 
  inputs=context, 
  outputs=output)
iface.launch(debug=True)

from sense2vec import Sense2Vec
s2v = Sense2Vec().from_disk('s2v_old')

from sentence_transformers import SentenceTransformer
sentence_transformer_model = SentenceTransformer('msmarco-distilbert-base-v3')

from similarity.normalized_levenshtein import NormalizedLevenshtein
normalized_levenshtein = NormalizedLevenshtein()

def filter_same_sense_words(original,wordlist):
  filtered_words=[]
  base_sense =original.split('|')[1] 
  print (base_sense)
  for eachword in wordlist:
    if eachword[0].split('|')[1] == base_sense:
      filtered_words.append(eachword[0].split('|')[0].replace("_", " ").title().strip())
  return filtered_words

def get_highest_similarity_score(wordlist,wrd):
  score=[]
  for each in wordlist:
    score.append(normalized_levenshtein.similarity(each.lower(),wrd.lower()))
  return max(score)

def sense2vec_get_words(word,s2v,topn,question):
    output = []
    print ("word ",word)
    try:
      sense = s2v.get_best_sense(word, senses= ["NOUN", "PERSON","PRODUCT","LOC","ORG","EVENT","NORP","WORK OF ART","FAC","GPE","NUM","FACILITY"])
      most_similar = s2v.most_similar(sense, n=topn)
      # print (most_similar)
      output = filter_same_sense_words(sense,most_similar)
      print ("Similar ",output)
    except:
      output =[]

    threshold = 0.6
    final=[word]
    checklist =question.split()
    for x in output:
      if get_highest_similarity_score(final,x)<threshold and x not in final and x not in checklist:
        final.append(x)
    
    return final[1:]


from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity

def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]
    for _ in range(top_n - 1):

        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (lambda_param) * candidate_similarities - (1-lambda_param) * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

def get_distractors_wordnet(word):
    distractors=[]
    try:
      syn = wn.synsets(word,'n')[0]
      
      word= word.lower()
      orig_word = word
      if len(word.split())>0:
          word = word.replace(" ","_")
      hypernym = syn.hypernyms()
      if len(hypernym) == 0: 
          return distractors
      for item in hypernym[0].hyponyms():
          name = item.lemmas()[0].name()
          #print ("name ",name, " word",orig_word)
          if name == orig_word:
              continue
          name = name.replace("_"," ")
          name = " ".join(w.capitalize() for w in name.split())
          if name is not None and name not in distractors:
              distractors.append(name)
    except:
      print ("Wordnet distractors not found")
    return distractors

def get_distractors (word,origsentence,sense2vecmodel,sentencemodel,top_n,lambdaval):
  distractors = sense2vec_get_words(word,sense2vecmodel,top_n,origsentence)
  print ("distractors ",distractors)
  if len(distractors) ==0:
    return distractors
  distractors_new = [word.capitalize()]
  distractors_new.extend(distractors)
  # print ("distractors_new .. ",distractors_new)

  embedding_sentence = origsentence+ " "+word.capitalize()
  # embedding_sentence = word
  keyword_embedding = sentencemodel.encode([embedding_sentence])
  distractor_embeddings = sentencemodel.encode(distractors_new)

  # filtered_keywords = mmr(keyword_embedding, distractor_embeddings,distractors,4,0.7)
  max_keywords = min(len(distractors_new),5)
  filtered_keywords = mmr(keyword_embedding, distractor_embeddings,distractors_new,max_keywords,lambdaval)
  # filtered_keywords = filtered_keywords[1:]
  final = [word.capitalize()]
  for wrd in filtered_keywords:
    if wrd.lower() !=word.lower():
      final.append(wrd.capitalize())
  final = final[1:]
  return final

context = gr.inputs.Textbox(lines=10, placeholder="Enter paragraph/content here...")
output = gr.outputs.HTML(  label="Question and Answers")
radiobutton = gr.inputs.Radio(["Wordnet", "Sense2Vec"])

def generate_question(context,radiobutton):
  summary_text = summarizer(context,summary_model,summary_tokenizer)
  for wrp in wrap(summary_text, 150):
    print (wrp)
  # np = getnounphrases(summary_text,sentence_transformer_model,3)
  np =  keywords(context,summary_text)
  np
  output=""
  for answer in np:
    ques = get_question(summary_text,answer,question_model,question_tokenizer)
    if radiobutton=="Wordnet":
      distractors = get_distractors_wordnet(answer)
    else:
      distractors = get_distractors(answer.capitalize(),ques,s2v,sentence_transformer_model,40,0.2)
    # output= output + ques + "\n" + "Ans: "+answer.capitalize() + "\n\n"
    output = output + "<b style='color:blue;'>" + ques + "</b>"
    # output = output + "<br>"
    output = output + "<b style='color:green;'>" + "Ans: " +answer.capitalize()+  "</b>"
    if len(distractors)>0:
      for distractor in distractors[:4]:
        output = output + "<b style='color:brown;'>" + distractor+  "</b>"
    output = output + "<br>"

  summary ="Summary: "+ summary_text
  for answer in np:
    summary = summary.replace(answer,"<b>"+answer+"</b>")
    summary = summary.replace(answer.capitalize(),"<b>"+answer.capitalize()+"</b>")
  output = output + "<p>"+summary+"</p>"
  return output

iface = gr.Interface(
  fn=generate_question, 
  inputs=[context,radiobutton], 
  outputs=output)
iface.launch(debug=True)