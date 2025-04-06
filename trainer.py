#run experiments for codenet
from my_model import Graphormer 

#from preprocess_codenet import get_spt_dataset
import sys
import argparse
import random
import pickle
import anytree
from anytree import AnyNode, RenderTree
from tqdm import tqdm
import torch
import torch.nn as nn
import dgl
import torchtext
from torchtext.data.utils import get_tokenizer
from process_util import my_CodeNet_data_batch_new,my_CodeNet_data_batch_old,my_CodeNet_data_batch_msg
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
# vocab = deserialize_vocab('E://workspace//my_project//data//processed_data//python_vocab.json')
vocab_size = len(vocab)

nclasses=10
print("##############开始读取数据！####################")
trainset,devset,testset=get_CodeNet_Dataset(path="", type="java",  device=device, vocab=vocab, Max_num_node = args.max_node_num)

model = Graphormer(num_classes=nclasses,token_vocabsize=vocab_size,num_encoder_layers=args.num_encoder_layers,num_attention_heads=args.num_attention_heads).to(device)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
    model.train()
    random.shuffle(trainset) 
    trainbar=tqdm(dataiterator(trainset,batch_size=args.batch_size),disable=args.nobar)
    print(trainbar)
    totalloss=0.0
    traincorrect=0
    for batch in trainbar:
        optimizer.zero_grad()
        print(batch)
        #my_CodeNet_data_batch(batch, device=None,Max_num_node=100, token_bert=None, model_bert=None)
        inputFeat_new, atten_mask_new,dists_new=my_CodeNet_data_batch_new(batch, device=device ,Max_num_node=args.max_node_num)
        inputFeat_old, atten_mask_old,dists_old=my_CodeNet_data_batch_old(batch, device=device ,Max_num_node=args.max_node_num)
        inputFeat_msg, atten_mask_msg=my_CodeNet_data_batch_msg(batch,device=device ,Max_length =args.max_node_num )
        #print("inputbatch",inputFeat.shape)
        #print("target",lables)
        #print(dists.shape)
        output_new = model(inputFeat_new, attn_mask=atten_mask_new, device=device, dist = dists_new)
        output_old = model(inputFeat_old, attn_mask=atten_mask_old, device=device, dist = dists_old)
        output_text = model(inputFeat_msg,attn_mask=atten_mask_msg, device=device)
        print("out:",output_new.shape)
        print("out:",output_old.shape)
        print('out:',output_text.shape)




        #output.reshape(-1)
        loss = criterion(output, lables)
        #print("loss",output.argmax(1))
        totalloss+=loss.item()
        #loss.backward()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        #optimizer.step()
        # 缩放损失并调用 backward()

        # 使用 GradScaler 步骤优化器
        scaler.step(optimizer)
        # 更新缩放器
        #print(output.argmax(1))
        scaler.update()
        scheduler.step()
        traincorrect+=(output.argmax(1) == lables).sum().item()
        del inputFeat, atten_mask, lables
        
    #print('avg loss:',totalloss/len(trainset)*args.batch_size)
    #print('train acc:',traincorrect/len(trainset))
    logger.info('avg loss:%s',totalloss/len(trainset)*args.batch_size)
    logger.info('train acc:%s',traincorrect/len(trainset))
    
    model.eval()
    with torch.no_grad():
        random.shuffle(devset) 
        devbar=tqdm(dataiterator(devset,batch_size=args.batch_size),disable=args.nobar)
        devtotal=len(devset)
        devcorrect=0
        for batch in devbar:
            inputFeat, atten_mask, lables, dists=my_CodeNet_data_batch(batch, device=device ,Max_num_node=args.max_node_num)
            output = model(inputFeat, attn_mask=atten_mask, device=device, dist = dists)
            devcorrect+=(output.argmax(1) == lables).sum().item()
            del inputFeat, atten_mask, lables

        #print('devacc:',devcorrect/devtotal)
        logger.info('devacc:%s',devcorrect/devtotal)

        random.shuffle(testset) 
        testbar=tqdm(dataiterator(testset,batch_size=args.batch_size),disable=args.nobar)
        testtotal=len(testset)
        #print("testset len:",testtotal)
        testcorrect=0
        for batch in testbar:
            inputFeat, atten_mask, lables, dists=my_CodeNet_data_batch(batch, device=device ,Max_num_node=args.max_node_num)
            output = model(inputFeat, attn_mask=atten_mask, device=device, dist = dists)
            #print("outs_test", output.argmax(1))
            testcorrect+=(output.argmax(1) == lables).sum().item()
            del inputFeat, atten_mask, lables
        #print('testacc:',testcorrect/testtotal)
        logger.info('testacc:%s',testcorrect/testtotal)

    if devcorrect/devtotal>=maxdevacc:
        maxdevacc=devcorrect/devtotal
        maxdevepoch=epoch
        #print('best epoch')
        logger.info('best epoch')
    if epoch-maxdevepoch>30:
        #print('early stop')
        #print('best epoch:',maxdevepoch)
        logger.info('early stop')
        logger.info('best epoch:%s',maxdevepoch)
        quit()
    
