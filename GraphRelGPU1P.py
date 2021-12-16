
import os
import json
import pickle
import spacy
import torch
import torch as T
import torch.nn as nn
from torch.nn.functional import log_softmax
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


#pydevd_pycharm.settrace('128.180.108.75 ', port=22, stdoutToServer=True, stderrToServer=True)
ARCH = '1p'

#path = '/Users/samanthachen/PycharmProjects/GraphRel/nytdata'
path = '/home/xuc321/project/nytdata'

raw_test_path = path + '/raw_test.json'
raw_train_path = path + '/raw_train.json'
raw_valid_path = path + '/raw_valid.json'

relation2id_path = path + '/relations2id.json'
# Load the data from source json.
with open(relation2id_path) as f:
    relation_ids = json.load(f)

test_path = path + '/sshdata/nytprocess100_test01.json'
train_path = path + '/sshdata/nytprocess100_train01.json'
valid_path = path + '/sshdata/nytprocess100_valid01.json'

def nyt_prepocess(path):
    with open(path) as f:
        idx, inp, pos, dep_fw, dep_bw, ans_ne, wgt_ne, ans_rel, wgt_rel = ([] for _ in range(9))
        idx_p, inp_p, pos_p, dep_fw_p, dep_bw_p, ans_ne_p, wgt_ne_p, ans_rel_p, wgt_rel_p = ([] for _ in range(9))

        for line in f.readlines():
            line = json.loads(line)
            if len(list(line['idx'])) <= 100:
                t = list(line['idx'])  # convert dict to list
                idx.append(torch.as_tensor(t))
                i = list(line['inp'])  # convert dict to list
                inp.append(torch.as_tensor(i))
                p = list(line['pos'])  # convert dict to list
                pos.append(torch.as_tensor(p))
                df = torch.as_tensor([i for i in line['dep_fw']])  # convert dict to list
                dep_fw.append(df)
                db = torch.as_tensor([i for i in line['dep_bw']])
                dep_bw.append(db)
                an = list(line['ans_ne'])
                ans_ne.append(torch.as_tensor(an))
                wn = list(line['wgt_ne'])
                wgt_ne.append(torch.as_tensor(wn))
                ar = torch.as_tensor([i for i in line['ans_rel']])
                ans_rel.append(ar)
                wr = torch.as_tensor([i for i in line['wgt_rel']])
                wgt_rel.append(wr)
        idx_p = pad_sequence(idx, batch_first=True,padding_value=0)
        inp_p = pad_sequence(inp, batch_first=True,padding_value=0)
        pos_p = pad_sequence(pos, batch_first=True)
        for i, c in enumerate(dep_fw):
            pad = torch.nn.ConstantPad2d((0, 100 - len(dep_fw[i]), 0, 100 - len(dep_fw[i])),0)
            dep_fw_p.append(pad(dep_fw[i]))
        dep_fw_p = pad_sequence(dep_fw_p, batch_first=True,padding_value=0)
        for i, c in enumerate(dep_bw):
            pad = torch.nn.ConstantPad2d((0, 100 - len(dep_bw[i]), 0, 100 - len(dep_bw[i])),0)
            dep_bw_p.append(pad(dep_bw[i]))
        dep_bw_p = pad_sequence(dep_bw_p, batch_first=True)

        ans_ne_p = pad_sequence(ans_ne, batch_first=True,padding_value=0)
        wgt_ne_p = pad_sequence(wgt_ne, batch_first=True,padding_value=-1)
        for i, c in enumerate(ans_rel):
            pad = torch.nn.ConstantPad2d((0, 100 - len(ans_rel[i]), 0, 100 - len(ans_rel[i])),0)
            ans_rel_p.append(pad(ans_rel[i]))
        ans_rel_p = pad_sequence(ans_rel_p, batch_first=True)
        for i, c in enumerate(wgt_rel):
            pad = torch.nn.ConstantPad2d((0, 100 - len(wgt_rel[i]), 0, 100 - len(wgt_rel[i])),-1)
            wgt_rel_p.append(pad(wgt_rel[i]))
        wgt_rel_p = pad_sequence(wgt_rel_p, batch_first=True)
        nyt = {'idx': idx_p, 'inp': inp_p, 'pos': pos_p, 'dep_fw': dep_fw_p ,
                      'dep_bw': dep_bw_p, 'ans_ne': ans_ne_p,'wgt_ne': wgt_ne_p,'ans_rel':ans_rel_p,'wgt_rel':wgt_rel_p}
    return nyt


print('start loading training data')
#train_tensor = nyt_prepocess(train_path)
train_tensor_path = path +  '/sshdata/nyt100_train_tensor.pt'
#torch.save(train_tensor, train_tensor_path )
train = torch.load(train_tensor_path)
#train = train_tensor
print('start loading validation data')
#valid_tensor = nyt_prepocess(valid_path)
valid_tensor_path = path +  '/sshdata/nyt100_valid_tensor.pt'
#torch.save(valid_tensor, valid_tensor_path )
valid = torch.load(valid_tensor_path)
#valid = valid_tensor
print('start loading test data')
#test_tensor = nyt_prepocess(test_path)
test_tensor_path = path +  '/sshdata/nyt100_test_tensor.pt'
#torch.save(test_tensor, test_tensor_path )
test = torch.load(test_tensor_path)
#test = test_tensor


NLP = spacy.load('en_core_web_lg')
POS = dict()
for pos in list(NLP.get_pipe('tagger').labels):
    POS[pos] = len(POS)+1
NUM_POS = len(POS) + 2 # 0 for NA
NUM_REL = len(relation_ids)
MXL = 100
'''Dataset for loading processed data into batch'''
class DS(Dataset):
    def __init__(self, idx_p, inp_p, pos_p, dep_fw_p, dep_bw_p, ans_ne_p, wgt_ne_p, ans_rel_p, wgt_rel_p):
        super(DS, self).__init__()
        self.idx = idx_p
        self.inp = inp_p
        self.pos = pos_p
        self.dep_fw = dep_fw_p
        self.dep_bw = dep_bw_p
        self.ans_ne = ans_ne_p
        self.wgt_ne = wgt_ne_p
        self.ans_rel = ans_rel_p
        self.wgt_rel = wgt_rel_p

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        return self.idx[index], self.inp[index], self.pos[index], self.dep_fw[index],self.dep_bw[index], self.ans_ne[index], self.wgt_ne[index],self.ans_rel[index], self.wgt_rel[index]
'''Create Dataloader to load data in batch'''

ld_tr = DataLoader(DS(train['idx'], train['inp'], train['pos'], train['dep_fw'], train['dep_bw'], train['ans_ne'], train['wgt_ne'],train['ans_rel'], train['wgt_rel']), batch_size=32, shuffle=True)

ld_vl = DataLoader(DS(valid['idx'], valid['inp'], valid['pos'], valid['dep_fw'], valid['dep_bw'], valid['ans_ne'], valid['wgt_ne'],valid['ans_rel'], valid['wgt_rel']), batch_size=64)

ld_ts = DataLoader(DS(test['idx'], test['inp'], test['pos'], test['dep_fw'], test['dep_bw'], test['ans_ne'], test['wgt_ne'],test['ans_rel'], test['wgt_rel']), batch_size=64)

'''GraphRel model frame'''
class GCN(nn.Module):
    def __init__(self, hid_size=256):
        super(GCN, self).__init__()

        self.hid_size = hid_size

        self.W = nn.Parameter(T.FloatTensor(self.hid_size, self.hid_size // 2).cuda())
        self.b = nn.Parameter(T.FloatTensor(self.hid_size // 2, ).cuda())

        self.init()

    def init(self):
        stdv = 1 / math.sqrt(self.hid_size // 2)

        self.W.data.uniform_(-stdv, stdv)
        # Fills self tensor with numbers sampled from the continuous uniform distribution(-stdv,stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, inp, adj, is_relu=True):
        """
        :param inp: [bs, ml, hid_size] (input)
        :param adj: [bs, ml, ml] (adjacency matrix)
        :param is_relu: True
        :return: [bs, ml, hid_size/2] output size will be half of the input size
        """
        out = T.matmul(inp, self.W) + self.b  # [bs, ml, hid_size/2]
        out = T.matmul(adj.float(), out)  # [bs, ml, hid_size/2]

        if is_relu == True:
            out = nn.functional.relu(out)

        return out  # [bs, ml, hid_size/2]

    def __repr__(self):
        return self.__class__.__name__ + '(hid_size=%d)' % (self.hid_size)
gcn = GCN().cuda()
print(gcn)

class Model_GraphRel(nn.Module):
    def __init__(self, mxl, num_rel,
                 hid_size=256, rnn_layer=2, gcn_layer=2, dp=0.5):
        super(Model_GraphRel, self).__init__()

        self.mxl = mxl  # max_seq_len
        self.num_rel = num_rel  # number of relations
        self.hid_size = hid_size  # hidden size of LSTM
        self.rnn_layer = rnn_layer
        self.gcn_layer = gcn_layer
        self.dp = dp

        self.emb_pos = nn.Embedding(NUM_POS, 15)

        self.rnn = nn.GRU(100 + 15, self.hid_size,
                          num_layers=self.rnn_layer, batch_first=True, dropout=dp,
                          bidirectional=True)  # Bi-GRU: hid_size *2
        self.gcn_fw = nn.ModuleList([GCN(self.hid_size * 2) for _ in range(self.gcn_layer)])  # forward gcn
        self.gcn_bw = nn.ModuleList([GCN(self.hid_size * 2) for _ in range(self.gcn_layer)])  # backward gcn
        # ne: named entity extraction RNN
        self.rnn_ne = nn.GRU(self.hid_size * 2, self.hid_size, batch_first=True)
        # (input:input size, h_0: hidden size),  Bi-GRU: hid_size * 2
        # batch_first = True. -> Input sequence(batch_size, seq_len, input_size)

        self.fc_ne = nn.Linear(self.hid_size, 5)  # based on names entity

        self.trs0_rel = nn.Linear(self.hid_size * 2, self.hid_size)  # linear layer based on relation extraction
        self.trs1_rel = nn.Linear(self.hid_size * 2, self.hid_size)
        self.fc_rel = nn.Linear(self.hid_size * 2, self.num_rel)  # based on relation extraction

        if ARCH == '2p':  # If it's on phrase 2 in GraphRel model
            # build forward gcn for each class of relation
            self.gcn2p_fw = nn.ModuleList([GCN(self.hid_size * 2) for _ in range(self.num_rel)])
            self.gcn2p_bw = nn.ModuleList([GCN(self.hid_size * 2) for _ in range(self.num_rel)])

        self.dp = nn.Dropout(dp)

    def output(self, feat):
        """
        :param feat:[bs, ml, hid_size*2]
        """
        # predict word entity
        out_ne, _ = self.rnn_ne(feat)  # [bs, ml, hid_size]
        out_ne = self.dp(out_ne)
        out_ne = self.fc_ne(out_ne)  # [bs, ml, 5]
        #
        # calculate relation tendency score
        trs0 = nn.functional.relu(self.trs0_rel(feat))  # [bs, ml, hid_size]
        trs0 = self.dp(trs0)
        trs1 = nn.functional.relu(self.trs1_rel(feat))  # [bs, ml, hid_size] non-linear transform
        trs1 = self.dp(trs1)

        trs0 = trs0.view((trs0.shape[0], trs0.shape[1], 1, trs0.shape[2]))  # [bs, ml, 1, hid_size]
        trs0 = trs0.expand((trs0.shape[0], trs0.shape[1], trs0.shape[1], trs0.shape[3]))  # [bs, ml, ml, hid_size]
        trs1 = trs1.view((trs1.shape[0], 1, trs1.shape[1], trs1.shape[2]))
        trs1 = trs1.expand((trs1.shape[0], trs1.shape[2], trs1.shape[2], trs1.shape[3]))
        # $S_{(w 1, r, w 2)}=W_{r}^{3} \operatorname{ReLU}\left(W_{r}^{1} h_{w 1} \oplus W_{r}^{2} h_{w 2}\right)$
        trs = T.cat([trs0, trs1], dim=3)  # [bs, ml, ml, hid_size*2]

        out_rel = self.fc_rel(trs)
        # [bs, ml, ml, num_rel]

        return out_ne, out_rel  # [bs, ml, 5] [bs, ml, ml, num_rel]

    def forward(self, inp, pos, dep_fw, dep_bw):
        """
        :param inp: [bs, ml, size] (input)
        :param pos: [bs, ml] (position)
        :param dep_fw: [bs, ml, ml] dependency tree forward
        :param dep_bw: [bs, ml, ml] dependency tree backward
        """
        pos = self.emb_pos(pos)  # [bs,ml,15] position embedding
        inp = T.cat([inp, pos], dim=2)  # [bs, ml,300+15] concatenate word embedding and pos
        inp = self.dp(inp)  # drop out
        out, _ = self.rnn(inp)  # [bs, ml, hid_size*2]

        for i in range(self.gcn_layer):
            out_fw = self.gcn_fw[i](out, dep_fw)  # [bs, ml, hs]
            out_bw = self.gcn_bw[i](out, dep_bw)
            out = T.cat([out_fw, out_bw], dim=2)  # [bs, ml,hs*2]
            out = self.dp(out)

        feat_1p = out  # [bs, ml,hs*2] #1st phrase feature
        out_ne, out_rel = self.output(feat_1p)  # [bs,ml,5], [bs,ml,ml, num_rel26] word entity, relation

        if ARCH == '1p':
            return out_ne, out_rel

        # 2p
        out_ne1, out_rel1 = out_ne, out_rel

        dep_fw = nn.functional.softmax(out_rel,
                                       dim=3)  # [bs, ml, ml, num_rel26] # n-dimensional input Tensor (index dim = 3)->rel
        dep_bw = dep_fw.transpose(1, 2)  # [bs,ml,ml, num_rel26]

        outs = []
        for i in range(self.num_rel):
            out_fw = self.gcn2p_fw[i](feat_1p, dep_fw[:, :, :, i])  # dep_fw: [bs,ml,ml]
            out_bw = self.gcn2p_bw[i](feat_1p, dep_bw[:, :, :, i])  # out_bw: [bs,ml,hid_size]
            outs.append(self.dp(T.cat([out_fw, out_bw], dim=2)))
            # [bs,ml, hs*2] * num_rel
        feat_2p = feat_1p  # [bs,ml,hs*2]
        for i in range(self.num_rel):
            feat_2p = feat_2p + outs[i]  # [bs,ml,hs*2]

        out_ne2, out_rel2 = self.output(feat_2p)  # [bs,ml,5] [bs,ml,ml,num_rel]

        return out_ne1, out_rel1, out_ne2, out_rel2


model = Model_GraphRel(mxl=MXL, num_rel=NUM_REL).cuda()

def post_proc(out_ne, out_rel):
    out_ne = np.argmax(out_ne.detach().cpu().numpy(), axis=1)
    out_rel = np.argmax(out_rel.detach().cpu().numpy(), axis=2)
    rels = triples(out_ne, out_rel)
    return rels
def triples(ne, rel):
    nes = dict()
    el = -1
    for i, v in enumerate(ne):
        if v == 4:  # single word in the entity
            nes[i] = [i, i]  # ml,ml
            el = -1

        elif v == 1:  # begin word in the entity
            el = i

        elif v == 3:  # end word in the entity
            if not el == -1:
                for p in range(el, i + 1):
                    nes[p] = [el, i]

        elif v == 2:  # v==3, middle word in the entity
            pass

        elif v == 0:  # word not entity
            el = -1

    rels = []
    for i in range(MXL):
        for j in range(MXL):
            if not rel[i][j] == 0 and i in nes and j in nes:
                rels.append([nes[i][1], nes[j][1], rel[i][j]])  # (entity 1, entity2, relation)

    cl = []
    for rel in rels:
        if not rel in cl:
            cl.append(rel)
    rels = cl
    return rels
class F1:
    def __init__(self):
        self.P = [0, 0]
        self.R = [0, 0]

    def get(self):
        try:
            P = self.P[0] / self.P[1]
        except:
            P = 0

        try:
            R = self.R[0] / self.R[1]
        except:
            R = 0

        try:
            F = 2 * P * R / (P + R)
        except:
            F = 0

        return P, R, F  # precision, recall , f1

    def add(self, ro, ra):
        self.P[1] += len(ro)  # count the number of TP + FP
        self.R[1] += len(ra)  # count the number of TP + FN

        for r in ro:
            if r in ra:
                self.P[0] += 1  # count TP

        for r in ra:  # count TP
            if r in ro:
                self.R[0] += 1


'''Experiment setting'''
EPOCHS = 200
LR = 0.0008
DECAY = 0.98

W_NE = 2
W_REL = 2
ALP = 3

'''CrossEntopy as loss function, ignore -1 as padding value'''
loss_func = nn.CrossEntropyLoss(reduction='none').cuda()
optim = T.optim.Adam(model.parameters(), lr=LR)
def ls(out_ne, wgt_ne, out_rel, wgt_rel):
    #criterion1 = nn.CrossEntropyLoss(reduction='mean', weight=wgt_ne).cuda()
    #criterion2 = nn.CrossEntropyLoss(reduction='mean', weight=wgt_rel).cuda()
    ls_ne = loss_func(out_ne.view((-1, 5)), ans_ne.view((-1,)).cuda())
    #loss_ne = ls_ne.sum()/wgt_ne[ans_ne.view((-1,))].sum().cuda()

    ls_ne = (ls_ne * wgt_ne.cuda()).sum() / (wgt_ne > 0).sum().cuda()

    ls_rel = loss_func(out_rel.view((-1, NUM_REL)), ans_rel.view((-1,)).cuda())
    ls_rel = (ls_rel * wgt_rel.cuda()).sum() / (wgt_rel > 0).sum().cuda()

    return ls_ne, ls_rel


'''Start model training'''
print("start training.............")
for e in tqdm(range(EPOCHS)):

    ls_ep_ne1, ls_ep_rel1 = 0, 0  # eloss1p:entity loss in 1st phrase, rloss1p: relation loss in 1st phrase


    model.train()
    with tqdm(ld_tr) as TQ:
        for i, (idx, inp, pos, dep_fw, dep_bw, ans_ne, wgt_ne, ans_rel, wgt_rel) in enumerate(TQ):

            wgt_ne.masked_fill_(wgt_ne == 1, W_NE)
            wgt_ne.masked_fill_(wgt_ne == 0, 1)
            wgt_ne.masked_fill_(wgt_ne == -1, 0)

            wgt_rel.masked_fill_(wgt_rel == 1, W_REL)
            wgt_rel.masked_fill_(wgt_rel == 0, 1)
            wgt_rel.masked_fill_(wgt_rel == -1, 0)

            out_ne1, out_rel1 = model(inp.cuda(), pos.cuda(), dep_fw.cuda(), dep_bw.cuda())
            ls_ne1, ls_rel1 = ls(out_ne1, wgt_ne.type(torch.FloatTensor), out_rel1, wgt_rel.type(torch.FloatTensor))

            optim.zero_grad()
            ((ls_ne1 + ls_rel1)).backward()
            optim.step()

            ls_ne1 = ls_ne1.detach().cpu().numpy()
            ls_rel1 = ls_rel1.detach().cpu().numpy()
            ls_ep_ne1 += ls_ne1
            ls_ep_rel1 += ls_rel1



            # TQ.set_postfix(ls_ne1='%.3f' % (ls_ne1), ls_rel1='%.3f' % (ls_rel1),
            #               ls_ne2='%.3f' % (ls_ne2), ls_rel2='%.3f' % (ls_rel2))

            # print('batch %d: ne1: %.4f, rel1: %.4f, ne2: %.4f, rel2: %.4f' % (i, ls_ne1, ls_rel1,
            #                                                           ls_ne2, ls_rel2))

            if i % 100 == 0:

                for pg in optim.param_groups:
                    pg['lr'] *= DECAY

        ls_ep_ne1 /= len(TQ)
        ls_ep_rel1 /= len(TQ)



        # print('Ep %d: ne1: %.4f, rel1: %.4f, ne2: %.4f, rel2: %.4f' % (e + 1, ls_ep_ne1, ls_ep_rel1,
        #                                                              ls_ep_ne2, ls_ep_rel2))
        print('Ep %d: ne1: %.4f, rel1: %.4f' % (e + 1, ls_ep_ne1, ls_ep_rel1))
        T.save(model.state_dict(), 'Model/nyt1P/%s_%s_%d.pt' % ('nyt', ARCH, e + 1))
    f1 = F1()
    model.eval()
    with tqdm(ld_vl) as TQ:
        for idx, inp, pos, dep_fw, dep_bw, ans_ne, wgt_ne, ans_rel, wgt_rel in TQ:

            out_ne, out_rel = model(inp.cuda(), pos.cuda(), dep_fw.cuda(), dep_bw.cuda())
            ans_ne = ans_ne.numpy()
            ans_rel = ans_rel.numpy()
            for i in range(idx.shape[0]):
                ans = triples(ans_ne[i], ans_rel[i])
                rels = post_proc(out_ne[i], out_rel[i])
                f1.add(rels, ans)
            #print('rels:', rels, 'ans:', ans)
        p, r, f = f1.get()
        print('P: %.4f%%, R: %.4f%%, F: %.4f%%' % (100 * p, 100 * r, 100 * f))
print("finish........")

