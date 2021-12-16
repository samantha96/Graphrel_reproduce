# Graphrel_reproduce
NYT dataset process 

   
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
