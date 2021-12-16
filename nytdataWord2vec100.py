import spacy
import json
import os
import pickle
import spacy
from spacy import displacy
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm_notebook as tqdm
import math
import unicodedata

#Normalize the unicode string.
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
    return index1, index2

def find_all_index(sen_split, word_split):
    """
    Find all loaction of entity in sentence.
    :param sen_split: the sentence array.
    :param word_split: the entity array.
    :return:
    index_list: the list of index pairs.
    """
    start, end, offset = -1, -1, 0
    # print("sen_split: ", sen_split)
    # print("word_split: ", word_split)
    index_list = []
    while (True):
        if len(index_list) != 0:
            offset = index_list[-1][1]
        start, end = find_index(sen_split[offset:], word_split)
        if start == -1 and end == -1: break
        if end <= start: break
        start += offset
        end += offset
        index_list.append((start, end))
    return index_list

# get word vectors from a file
# use index as key and dependency matrix as value

raw_test_path = '/Users/samanthachen/PycharmProjects/GraphRel/nytdata/raw_test.json'
raw_train_path = '/Users/samanthachen/PycharmProjects/GraphRel/nytdata/raw_train.json'
raw_valid_path = '/Users/samanthachen/PycharmProjects/GraphRel/nytdata/raw_valid.json'
output_path_test = '/Users/samanthachen/PycharmProjects/GraphRel/nytdata/nytprocess100_test.json'
output_path_train = '/Users/samanthachen/PycharmProjects/GraphRel/nytdata/nytprocess100_train.json'
output_path_valid = '/Users/samanthachen/PycharmProjects/GraphRel/nytdata/nytprocess100_valid.json'

relation2id_path = '/Users/samanthachen/PycharmProjects/GraphRel/nytdata/relations2id.json'
words2id_path = '/Users/samanthachen/PycharmProjects/GraphRel/nytdata/words2id.json'
words_id2vector = '/Users/samanthachen/PycharmProjects/GraphRel/nytdata/words_id2vector.json'


with open(words2id_path) as f:
    word2id = json.load(f)

with open(words_id2vector) as f:
    wordvec = json.load(f)

words_vectors = [[0] * 100 for i in range(len(wordvec))]
for id, vec in wordvec.items():
    words_vectors[int(id)] = vec

# Load the data from source json.
with open(relation2id_path) as f:
    relation_ids = json.load(f)

def readdata(raw_path):
    with open(raw_path) as f:
        nyt = []
        for line in f.readlines():
            line = json.loads(line)
            nyt.append(line)
    return nyt
nyt_test = readdata(raw_test_path)
nyt_train = readdata(raw_train_path)
nyt_valid = readdata(raw_valid_path)
'''
with open(raw_test_path) as f:
    nyt_test = []
    for line in f.readlines():
        line = json.loads(line)
        nyt_test.append(line)
with open(raw_train_path) as f:
    nyt_train = []
    for line in f.readlines():
        line = json.loads(line)
        nyt_train.append(line)

with open(raw_valid_path) as f:
    nyt_valid = []
    for line in f.readlines():
        line = json.loads(line)
        nyt_valid.append(line)
'''

""" {"sentText": "The extra money came because New Jersey is trying to be more sophisticated in how it assesses risk , said Jack Burns , the coordinator of emergency management in Hudson , where Jersey City is located . ''", 
    "articleId": "/m/vinci8/data1/riedel/projects/relation/kb/nyt1/docstore/nyt-2005-2006.backup/1784095.xml.pb", 
    "relationMentions": [{"em1Text": "New Jersey", "em2Text": "Jersey City", "label": "/location/location/contains"}], 
    "entityMentions": [{"start": 0, "label": "LOCATION", "text": "New Jersey"}, 
                       {"start": 1, "label": "PERSON", "text": "Jack Burns"}, 
                       {"start": 2, "label": "LOCATION", "text": "Hudson"}, 
                       {"start": 3, "label": "LOCATION", "text": "Jersey City"}], "sentId": "2"}"""


def proc(raw_path,output_path):
    nyt = readdata(raw_path)
    fw = open(output_path, "w+", encoding="utf-8")
    nlp = spacy.load('en_core_web_lg')
    parser = dict()
    for pars in list(nlp.parser.labels):
        parser[pars] = len(parser) + 1

    tagger = dict()
    for pos in list(nlp.tagger.labels):
        tagger[pos] = len(tagger) + 1
    for data in nyt:
        EpochCount = 0
        EpochCount += 1
        text_tag = {}
        sentText = normalize_text(data["sentText"]).rstrip('\n').rstrip("\r")
        sentDoc = nlp(sentText)
        sentWords = [token.text for token in sentDoc]
        text_tag["idx"] = [token.i for token in sentDoc]
        # "input" (BS,SL,ED）（Batch_size,sentence_length,embedding_dimension) here(SL,ED）
        text_tag["inp"] = []
        # get word2vec embedding for each word in each sentence
        # wordvec[word2id["Pronk"]]
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
            # dep_fw:     [seq_length, seq_length].       The dependencty adjacency matrix (forward edge) of each word-pair.
            # dep_bw:     [seq_length, seq_length].       The dependencty adjacency matrix (backward edge) of each word-pair.
            # By using the index of tokens with respect to the head,
            # you will get the distance to the head in the raw sentence rather than the dependency distance.

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
        for i, c in enumerate(sentWords):
            word = sentWords[i]
            if word in word2id.keys():  # incasecouldnotfindwordidinword2id
                id = word2id[word]
        # text_tag["inp"].append(list(nlp.vocab[word].vector))#300dimensionusing spacy(word2vec)
                text_tag["inp"].append(words_vectors[id])  # 100dimensionusingpre-trainedword3vec
            else:
                text_tag["inp"].append([0] * 100)
        '''
         get entity label, only considering BIOES label.
            B: begin word of an entity -> 2
            I: inner word of an entity -> 3
            E: end word of an entity   -> 4
            S: this word is a single-word entity  -> 1
            O: this word does not belong to entity -> 0
        '''
        text_tag["ans_ne"] = [0] * len(sentWords)
        for entity in data["entityMentions"]:
            entity_doc = nlp(normalize_text(entity["text"]))
            entity_list = [token.text for token in entity_doc]
            entity_idxs = find_all_index(sentWords, entity_list)

            for index in entity_idxs:
                if index[1] - index[0] == 1:
                    text_tag["ans_ne"][index[0]] = 4  # single
                elif index[1] - index[0] == 2:
                    text_tag["ans_ne"][index[0]] = 1  # begin
                    text_tag["ans_ne"][index[1] - 1] = 3  # end
                elif index[1] - index[0] > 2:
                    for i in range(index[0], index[1]):
                        text_tag["ans_ne"][i] = 2  # middle
                        text_tag["ans_ne"][index[0]] = 1 # begin
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
            entity1_idxs = find_all_index(sentWords, entity1_list)
            entity2_idxs = find_all_index(sentWords, entity2_list)

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

        if EpochCount % 10000 == 0:
            print("Epoch ", EpochCount)

        fw.write(json.dumps(text_tag, ensure_ascii=False) + '\n')

    fw.close()

proc(raw_train_path,output_path_train)
proc(raw_valid_path,output_path_valid)
proc(raw_test_path,output_path_test)

test_path = output_path_test
train_path =output_path_train
valid_path =output_path_valid

def nyt_prepocess(path):
    with open(path) as f:
        idx, inp, pos, dep_fw, dep_bw, ans_ne, wgt_ne, ans_rel, wgt_rel = ([] for _ in range(9))
        idx_p, inp_p, pos_p, dep_fw_p, dep_bw_p, ans_ne_p, wgt_ne_p, ans_rel_p, wgt_rel_p = ([] for _ in range(9))

        for line in f.readlines():
            line = json.loads(line)
            if len(list(line['idx'])) <= 100:
                t = list(line['idx'])  # convert dict to list
                idx.append(torch.tensor(t))
                i = list(line['inp'])  # convert dict to list
                inp.append(torch.tensor(i))
                p = list(line['pos'])  # convert dict to list
                pos.append(torch.tensor(p))
                df = torch.tensor([i for i in line['dep_fw']])  # convert dict to list
                dep_fw.append(df)
                db = torch.tensor([i for i in line['dep_bw']])
                dep_bw.append(db)
                an = list(line['ans_ne'])
                ans_ne.append(torch.tensor(an))
                wn = list(line['wgt_ne'])
                wgt_ne.append(torch.tensor(wn))
                ar = torch.tensor([i for i in line['ans_rel']])
                ans_rel.append(ar)
                wr = torch.tensor([i for i in line['wgt_rel']])
                wgt_rel.append(wr)
        idx_p = torch.nn.utils.rnn.pad_sequence(idx, batch_first=True)
        inp_p = torch.nn.utils.rnn.pad_sequence(inp, batch_first=True)
        pos_p = torch.nn.utils.rnn.pad_sequence(pos, batch_first=True)
        for i, c in enumerate(dep_fw):
            pad = torch.nn.ZeroPad2d(padding=(0, 100 - len(dep_fw[i]), 0, 100 - len(dep_fw[i])))
            dep_fw_p.append(pad(dep_fw[i]))
        dep_fw_p = torch.nn.utils.rnn.pad_sequence(dep_fw_p, batch_first=True)
        for i, c in enumerate(dep_bw):
            pad = torch.nn.ZeroPad2d(padding=(0, 100 - len(dep_bw[i]), 0, 100 - len(dep_bw[i])))
            dep_bw_p.append(pad(dep_bw[i]))
        dep_bw_p = torch.nn.utils.rnn.pad_sequence(dep_bw_p, batch_first=True)

        ans_ne_p = torch.nn.utils.rnn.pad_sequence(ans_ne, batch_first=True)
        wgt_ne_p = torch.nn.utils.rnn.pad_sequence(wgt_ne, batch_first=True)
        for i, c in enumerate(ans_rel):
            pad = torch.nn.ZeroPad2d(padding=(0, 100 - len(ans_rel[i]), 0, 100 - len(ans_rel[i])))
            ans_rel_p.append(pad(ans_rel[i]))
        ans_rel_p = torch.nn.utils.rnn.pad_sequence(ans_rel_p, batch_first=True)
        for i, c in enumerate(wgt_rel):
            pad = torch.nn.ZeroPad2d(padding=(0, 100 - len(wgt_rel[i]), 0, 100 - len(wgt_rel[i])))
            wgt_rel_p.append(pad(wgt_rel[i]))
        wgt_rel_p = torch.nn.utils.rnn.pad_sequence(wgt_rel_p, batch_first=True)
        nyt_tensor = {'idx': idx_p, 'inp': inp_p, 'pos': pos_p, 'dep_fw': dep_fw_p ,
                      'dep_bw': dep_bw_p, 'ans_ne': ans_ne_p,'wgt_ne': wgt_ne_p,'ans_rel':ans_rel_p,'wgt_rel':wgt_rel_p}
    return nyt_tensor

print('start loading training data')
train_tensor = nyt_prepocess(train_path)
train_tensor_path = '/Users/samanthachen/PycharmProjects/GraphRel/nytdata/nyt100_train_tensor.pt'
torch.save(train_tensor, train_tensor_path )
train = torch.load(train_tensor_path)

print('start loading validation data')
valid_tensor = nyt_prepocess(valid_path)
valid_tensor_path = '/Users/samanthachen/PycharmProjects/GraphRel/nytdata/nyt100_valid_origin_tensor.pt'
torch.save(valid_tensor, valid_tensor_path )
valid = torch.load(valid_tensor_path)

print('start loading test data')
test_tensor = nyt_prepocess(test_path)
test_tensor_path = '/Users/samanthachen/PycharmProjects/GraphRel/nytdata/nyt100_test_tensor.pt'
torch.save(test_tensor, test_tensor_path )
test = torch.load(test_tensor_path)








