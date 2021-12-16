'''
This is the python script for pre-process NYT raw dataset.
Here is a sample of NYT raw dataset:
{"sentText": "In Queens , North Shore Towers , near the Nassau border , supplanted a golf course , and housing replaced a gravel quarry in Douglaston .",
"articleId": "/m/vinci8/data1/riedel/projects/relation/kb/nyt1/docstore/nyt-2005-2006.backup/1719412.xml.pb",
"relationMentions": [{"em1Text": "Douglaston", "em2Text": "Queens", "label": "/location/neighborhood/neighborhood_of"},
                     {"em1Text": "Queens", "em2Text": "Douglaston", "label": "/location/location/contains"}],
"entityMentions": [{"start": 0, "label": "LOCATION", "text": "Queens"},
                   {"start": 1, "label": "LOCATION", "text": "Nassau"},
                   {"start": 2, "label": "LOCATION", "text": "Douglaston"}], "sentId": "2"}
for idx, inp, pos, dep_fw, dep_bw, ans_ne, wgt_ne, ans_rel, wgt_rel in ld_ts
Here is the data format after run this script：
(BS: batch_size
SL: sentence_length
ED: embedding_dimension)

idx: not important, not used
inp: the input sentence ([BS, SL, ED])
pos: the part-of-speech tag of each word ([BS, SL])
dep_fw: the dependencty adjacency matrix (forward edge) of each word-pair ([BS, SL, SL])
dep_bw: the dependencty adjacency matrix (backward edge) of each word-pair ([BS, SL, SL])

ans_ne, ans_rel: the output tag of name entity of each word and relation of each word-pair ([BS, SL] ans [BS, SL, SL])
wgt_ne, wgt_rel: the loss weight of name entity of each word and relation of each word-pair, 1 for those contains name entity or relation, otherwise 0 ([BS, SL], [BS, SL, SL])
'''
import os
import json
import pickle
import spacy
import torch
import torch as T
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.notebook import tqdm as tqdm
import math
import unicodedata

#ARCH = '1p'

path = '/Users/samanthachen/PycharmProjects/GraphRel/nytdata'
#path = '/home/xuc321/project/nytdata'

raw_test_path = path + '/raw_test.json'
raw_train_path = path + '/raw_train.json'
raw_valid_path = path + '/raw_valid.json'
relation2id_path = path + '/relations2id.json'
words2id_path = path + '/words2id.json'
words_id2vector = path + '/words_id2vector.json'

test_path = path + '/nytproc_test.json'
train_path = path + '/nytproc_train.json'
valid_path = path + '/nytproc_valid.json'
relation_ids = json.load(open(relation2id_path))
word2id = json.load(open(words2id_path))
wordvec = json.load(open(words_id2vector) )
words_vectors = [[0] * 100 for i in range(len(wordvec))]
for id, vec in wordvec.items():
    words_vectors[int(id)] = vec

def readdata(raw_path):
    with open(raw_path) as f:
        nyt = []
        for line in f.readlines():
            line = json.loads(line)
            nyt.append(line)
    return nyt

def normalize_text(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

def find_index(sen_split, word_split):
    """
    Find the loaction of entity in sentence.
    :param sen_split: the sentence array
    :param word_split: the entity array
    :return:
    index 1: start index
    index 2: end index
    """
    index1 = -1
    index2 = -1
    for i in range(len(sen_split)):
        if str(sen_split[i]) == str(word_split[0]):
            flag = True
            k = i
            for j in range(len(word_split)):
                if word_split[j] != sen_split[k]:
                    flag = False
                if k < len(sen_split) - 1:
                    k += 1
            if flag:
                index1 = i
                index2 = i + len(word_split)
                break
    index_list = []
    index_list.append((index1, index2))
    return index_list

nlp = spacy.load('en_core_web_lg')
'''parser -> list the dependency tree label '''
parser = dict()
for pars in list(nlp.get_pipe('parser').labels):
    parser[pars] = len(parser) + 1
'''tagger -> list the of part of speech label'''
tagger = dict()
for pos in list(nlp.get_pipe('tagger').labels):
    tagger[pos] = len(tagger) + 1

def proc(raw_path, output_path):
    nyt = readdata(raw_path)
    fw = open(output_path, "w+", encoding="utf-8")
    for data in nyt:
        text_tag = {}
        sentText = normalize_text(data["sentText"]).rstrip('\n').rstrip("\r")
        sentDoc = nlp(sentText)
        sentWords = [token.text for token in sentDoc]
        text_tag["idx"] = [token.i for token in sentDoc]
        # "input" (BS,SL,ED）（Batch_size,sentence_length,embedding_dimension) here(SL,ED）
        text_tag["inp"] = []
        # get word2vec embedding for each word in each sentence
        # pos: the part-of-speech tag of each word ([BS, SL])
        text_tag["pos"] = []
        # dep_fw: the dependencty adjacency matrix (forward edge) of each word-pair ([BS, SL, SL])
        text_tag["dep_fw"] = [[0] * len(sentWords) for i in range(len(sentWords))]
        # dep_bw: the dependencty adjacency matrix (backward edge) of each word-pair ([BS, SL, SL])
        text_tag["dep_bw"] = [[0] * len(sentWords) for i in range(len(sentWords))]
        # for token in sentDoc:
        #    print(token.text, token.pos_, token.dep_)
        for token in sentDoc:
            # pos:        [seq_length].
            ''' get the part-of-speech tag of each word.'''
            text_tag["pos"].append(tagger.get(token.tag_))
            # len(text_tag["pos"])  #37
            # dep_fw: [seq_length, seq_length]. The dependency adjacency matrix (forward edge) of each word-pair.
            # dep_bw: [seq_length, seq_length]. The dependency adjacency matrix (backward edge) of each word-pair.
            ''' get dependency parser adjacency matrix'''
            if token.i >= token.head.i:
                text_tag["dep_fw"][token.i][token.head.i] = parser.get(token.dep_)
            else:
                text_tag["dep_bw"][token.i][token.head.i] = parser.get(token.dep_)

        ''' get word embedding 300 dimension
        for i, c in enumerate(sentWords):
            word = sentWords[i]
            text_tag["inp"].append(nlp(word).vector.tolist())
        '''
        ''' get word embedding 100 dimension'''
        # text_tag["inp"].append(list(nlp.vocab[word].vector))#300dimensionusingspacy(word2vec)
        for i, c in enumerate(sentWords):
            word = sentWords[i]
            if word in word2id.keys():  # incase could not find wordid in word2id
                id = word2id[word]
                text_tag["inp"].append(words_vectors[id])  # 100 dimension
            else:
                text_tag["inp"].append([0] * 100)
        '''
         get entity label, only considering BIOES label.
            B: begin word of an entity -> 1
            I: inner word of an entity -> 2
            E: end word of an entity   -> 3
            S: this word is a single-word entity  -> 4
            O: this word does not belong to entity -> 0
        '''
        text_tag["ans_ne"] = [0] * len(sentWords)
        for entity in data["entityMentions"]:
            entity_doc = nlp(normalize_text(entity["text"]))
            entity_list = [token.text for token in entity_doc]
            entity_idxs = find_index(sentWords, entity_list)

            for index in entity_idxs:
                if index[1] - index[0] == 1:
                    text_tag["ans_ne"][index[0]] = 4  # single
                elif index[1] - index[0] == 2:
                    text_tag["ans_ne"][index[0]] = 1  # begin
                    text_tag["ans_ne"][index[1] - 1] = 3  # end
                elif index[1] - index[0] > 2:
                    for i in range(index[0], index[1]):
                        text_tag["ans_ne"][i] = 2  # middle
                        text_tag["ans_ne"][index[0]] = 1  # begin
                        text_tag["ans_ne"][index[1] - 1] = 3  # end

        text_tag["wgt_ne"] = [0] * len(sentWords)
        for i, c in enumerate(text_tag["ans_ne"]):
            if c != 0:
                text_tag["wgt_ne"][i] = 1
        # ans_rel:    [seq_length, seq_length].       The output relation of each word-pair.
        # relationMentions:               The gold relational triples.
        text_tag["ans_rel"] = [[0] * len(sentWords) for i in range(len(sentWords))]
        for relation in data["relationMentions"]:
            entity1_list = [token.text for token in nlp(normalize_text(relation["em1Text"]))]
            entity2_list = [token.text for token in nlp(normalize_text(relation["em2Text"]))]
            entity1_idxs = find_index(sentWords, entity1_list)
            entity2_idxs = find_index(sentWords, entity2_list)

            for en1_idx in entity1_idxs:
                for en2_idx in entity2_idxs:
                    for i in range(en1_idx[0], en1_idx[1]):
                        for j in range(en2_idx[0], en2_idx[1]):
                            text_tag["ans_rel"][i][j] = relation_ids[
                                relation["label"]]  # relation_ids['/location/location/contains']  #22

        text_tag["wgt_rel"] = [[0] * len(sentWords) for i in range(len(sentWords))]
        for i, c in enumerate(text_tag["ans_rel"]):
            for j, d in enumerate(c):
                if d != 0:
                    text_tag["wgt_rel"][i][j] = 1
            """relation_item = {}
            if normalize_text(relation["label"]) != "None":
                relation_item["label"] = relation["label"]
                relation_item["label_id"] = relation_ids[relation["label"]]
                relation_item["em1Text"] = entity1_list
                relation_item["em2Text"] = entity2_list
                text_tag["relationMentions"].append(relation_item)"""
        fw.write(json.dumps(text_tag, ensure_ascii=False))

    fw.close()

proc(raw_train_path,train_path)
proc(raw_valid_path, valid_path)
proc(raw_test_path,test_path)
