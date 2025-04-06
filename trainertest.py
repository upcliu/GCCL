#run experiments for codenet
from my_model import Graphormer,EncoderText1,TextSA,EncoderSimilarity,ContrastiveLoss

#from preprocess_codenet import get_spt_dataset
import sys
import argparse
import random
import pickle
import numpy as np
import anytree
from anytree import AnyNode, RenderTree
from tqdm import tqdm
import torch
import torch.nn as nn
import dgl
import torchtext
from torchtext.data.utils import get_tokenizer
from process_util import my_CodeNet_data_batch_new,my_CodeNet_data_batch_old,my_CodeNet_data_batch_msg,my_CodeNet_data_batch_lenth
from utils import dataiterator,NoamLR
#from treetransformernew import TreeTransformer_typeandtoken
from get_CodeNet2 import get_CodeNet_Dataset
import time
from torch.cuda.amp import GradScaler, autocast
import json
import os
os.environ['HF_ENDPOINT'] ='https://hf-mirror.com'
from transformers import RobertaModel, RobertaTokenizer, AutoModel, AutoTokenizer
from tree_sitter import Language, Parser
import torch
from get_vocab import Vocabulary, deserialize_vocab
from logger_util import  get_logger_for_handler_console

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='java250', help="dataset name")
parser.add_argument("--emsize", type=int, default=256, help="embedding dim")
parser.add_argument("--num_heads", type=int, default=8, help="attention heads")
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--maxepoch", type=int, default=200, help="max training epochs")
parser.add_argument("--nobar", type=boolean_string, default=True, help="disable progress bar")
parser.add_argument("--num_attention_heads", type=int, default=3, help="heads")#头数
parser.add_argument("--max_node_num", type=int, default=256, help="最大节点个数")
parser.add_argument("--num_encoder_layers", type=int, default=3, help="graphormer layers")

parser.add_argument("--log_path", type=str, default="log.txt", help="log file path")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = AutoTokenizer.from_pretrained("microsoft/codebert-base")

vocab_size = len(vocab)

nclasses=10
print("##############开始读取数据！####################")
trainset,devset,testset=get_CodeNet_Dataset(path="", type="java",  device=device, vocab=vocab, Max_num_node = args.max_node_num)

text_model = EncoderText1(vocab_size=vocab_size, word_dim=512, embed_size=512, num_layers=6, no_txtnorm=True)
get_global = TextSA(embed_dim = 512,dropout_rate = 0.1)
optimizer = torch.optim.Adam(text_model.parameters(), lr=args.lr)
optimizer_get_global = torch.optim.Adam(get_global.parameters(), lr=args.lr)
code_model = Graphormer( embed_size=512,token_vocabsize=vocab_size,num_encoder_layers=3,num_attention_heads=3).to(device)
optimizer_code_model = torch.optim.Adam(code_model.parameters(), lr=args.lr)
Similarity = EncoderSimilarity(embed_size=512, sim_dim=256, module_name='SGR', sgr_step=5)
optimizer_code_model = torch.optim.Adam(Similarity.parameters(), lr=args.lr)
loss_fn = ContrastiveLoss(margin=0.2, max_violation=False)
#warmup_steps是warmup的步数，scheduler是学习率衰减策略
warmup_steps=2000
scheduler=NoamLR(optimizer,warmup_steps=warmup_steps)
scaler = GradScaler()
print('max epoch:',args.maxepoch)
maxdevacc=0
maxdevepoch=0

logger = get_logger_for_handler_console(args.log_path)
logger.info('max epoch:%s',args.maxepoch)
for epoch in range(args.maxepoch):
    print("####################################################################")
    print('epoch:',epoch+1)
    sys.stdout.flush()
    text_model.train()
    random.shuffle(trainset) 
    trainbar=tqdm(dataiterator(trainset,batch_size=args.batch_size),disable=args.nobar)
    print(trainbar)
    totalloss=0.0
    traincorrect=0
    for batch in trainbar:
        optimizer.zero_grad()
        #print(batch)
        inputFeat_msg, atten_mask_msg=my_CodeNet_data_batch_msg(batch,device=device ,Max_length =args.max_node_num )
        inputFeat_old, atten_mask_old,dists_old=my_CodeNet_data_batch_old(batch, device=device ,Max_num_node=args.max_node_num)
        inputFeat_new, atten_mask_new,dists_new=my_CodeNet_data_batch_new(batch, device=device ,Max_num_node=args.max_node_num)
        rel_lengths = my_CodeNet_data_batch_lenth(batch, device=device)
        # print(rel_lengths.shape)
        output_text = text_model(inputFeat_msg) # shape (batch_size, seq_len, embed_size) 69, 256, 512
        output_code_new = code_model(inputFeat_new, attn_mask=atten_mask_new, device=device, dist = dists_new)
        output_code_old = code_model(inputFeat_old, attn_mask=atten_mask_old, device=device, dist = dists_old)
        
        # print(rel_lengths)
        # print(output_code_old.shape)
        #print(output_text.shape)   
        raw_global_text = torch.mean(output_text, axis=1)
        new_global_text = get_global(output_text,raw_global_text)
        output_code_local = torch.randn(69, 36, 512) #  待定  这个地方主要是局部的变更信息。
        similarity_matrix = Similarity(output_code_local,output_code_old,output_text,rel_lengths)

        loss = loss_fn(similarity_matrix)
        print(loss)
           #print("loss",output.argmax(1))
        totalloss+=loss.item()
        #loss.backward()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(code_model.parameters(), 0.5)
        #optimizer.step()
        # 缩放损失并调用 backward()

        # 使用 GradScaler 步骤优化器
        scaler.step(optimizer)
        # 更新缩放器
        #print(output.argmax(1))
        scaler.update()
        scheduler.step()
        del inputFeat_msg
