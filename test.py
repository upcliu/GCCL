#run experiments for codenet
from my_model import Graphormer,EncoderText1,TextSA,EncoderSimilarity

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
vocab = AutoTokenizer.from_pretrained("microsoft/codebert-base")
vocab_size = len(vocab)
""" x = torch.ones(256, 512, dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Graphormer( embed_size=512,token_vocabsize=vocab_size,num_encoder_layers=3,num_attention_heads=3).to(device)
output = model(x)
print(output.shape) """
encoder = EncoderSimilarity(embed_size=1024, sim_dim=256, module_name='SGR', sgr_step=5)

img_emb = torch.randn(32, 36, 1024)  # 假设图像特征的形状为(batch_size, 36, 1024)
code_global_emb = torch.randn(32, 1024)  # 假设代码全局特征的形状为(batch_size, 1024)
cap_emb = torch.randn(32, 20, 1024)  # 假设句子特征的形状为(batch_size, L, 1024)
cap_lens = torch.randint(1, 20, (32,))  # 假设句子长度为(batch_size,)
print(cap_lens.shape)
similarity_matrix = encoder(img_emb, code_global_emb, cap_emb, cap_lens)

print(similarity_matrix.shape)