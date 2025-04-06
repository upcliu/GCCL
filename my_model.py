import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import DegreeEncoder, GraphormerLayer, PathEncoder, SpatialEncoder


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = th.abs(X).sum(dim=dim, keepdim=True) + eps
    X = th.div(X, norm)
    return X


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = th.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = th.div(X, norm)
    return X

class EncoderText1(nn.Module):
    def __init__(self, vocab_size, word_dim, embed_size, num_layers, no_txtnorm=True):  #embed_size 256 
        super(EncoderText1, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # Word embedding layer
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.init_weights()
        self.dropout = nn.Dropout(0.4)

        # Transformer Encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(word_dim, embed_size)

    def init_weights(self):
        nn.init.kaiming_uniform_(self.embed.weight, nonlinearity='relu')

    def forward(self, captions):
        cap_emb = self.embed(captions)
        cap_emb = self.fc(cap_emb)
        cap_emb = self.dropout(cap_emb)

        # Transformer Encoder expects input of shape (seq_len, batch_size, embed_size)
        cap_emb = cap_emb.transpose(0, 1)
        # Pass through Transformer Encoder
        cap_emb = self.transformer_encoder(cap_emb)
        # Restore original shape (batch_size, seq_len, embed_size)
        cap_emb = cap_emb.transpose(0, 1)
        # normalization in the joint embedding space
        return cap_emb  # size(386, 128, 512)

class Graphormer(nn.Module):
    def __init__(
        self,
        embed_size, #num_classes是分类的类别数
        token_vocabsize = 0,
        edge_dim=3,       #edge_dim是边特征的维度
        num_atoms=4608,   #num_atoms是原子特征的维度
        max_degree=512,
        num_spatial=511,
        multi_hop_max_dist=5,  #multi_hop_max_dist是多跳的最大距离
        num_encoder_layers=3,
        embedding_dim=768,
        ffn_embedding_dim=768,
        num_attention_heads=3,
        dropout=0.1,
        pre_layernorm=True,
        activation_fn=nn.GELU(),
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim
        self.num_heads = num_attention_heads
        self.token_embeddings=nn.Embedding(token_vocabsize,embedding_dim)
        self.atom_encoder = nn.Embedding(
            num_atoms + 1, embedding_dim, padding_idx=0
        )
        self.graph_token = nn.Embedding(1, embedding_dim)

        self.degree_encoder = DegreeEncoder(
            max_degree=max_degree, embedding_dim=embedding_dim
        )

        self.path_encoder = PathEncoder(
            max_len=multi_hop_max_dist,
            feat_dim=edge_dim,
            num_heads=num_attention_heads,
        )

        self.spatial_encoder = SpatialEncoder(
            max_dist=num_spatial, num_heads=num_attention_heads
        )
        self.graph_token_virtual_distance = nn.Embedding(1, num_attention_heads)

        self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.main_layer = GraphormerLayer(
                    feat_size=self.embedding_dim,
                    hidden_size=ffn_embedding_dim,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    activation=activation_fn,
                    norm_first=pre_layernorm,
                )
        # map graph_rep to num_classes
        self.lm_head_transform_weight = nn.Linear(
            self.embedding_dim, self.embedding_dim
        )
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.activation_fn = activation_fn
        self.embed_out = nn.Linear(self.embedding_dim, embed_size, bias=False)
        self.lm_output_learned_bias = nn.Parameter(th.zeros(embed_size))

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(th.zeros(1))
        self.embed_out.reset_parameters()

    def forward(self, x, attn_mask=None, device=None, dist=None):
        
        batch, max_num_nodes = x.shape
        x = self.token_embeddings(x)
        if dist==None:
            attn_bias = th.zeros(
            batch,
            max_num_nodes,
            max_num_nodes,
            self.num_heads,
            device=device,
            )
            x = self.emb_layer_norm(x)
            x = self.main_layer(x, attn_mask=attn_mask, attn_bias=attn_bias)
            x = self.dropout(x)
            graph_rep = x[:, 0, :]
            graph_rep = self.layer_norm(
                self.activation_fn(self.lm_head_transform_weight(graph_rep))
            )
            graph_rep = self.embed_out(graph_rep)  
        else:
            attn_bias = th.zeros(
                batch,
                max_num_nodes,
                max_num_nodes,
                self.num_heads,
                device=dist.device,
            )
            spatial_encod = self.spatial_encoder(dist)
            x = self.emb_layer_norm(x)
            x = self.main_layer(x, attn_mask=attn_mask, attn_bias=spatial_encod)
            x = self.dropout(x)
            graph_rep = x[:, 0, :]
            graph_rep = self.layer_norm(
                self.activation_fn(self.lm_head_transform_weight(graph_rep))
            )
            graph_rep = self.embed_out(graph_rep)
        return graph_rep


class TextSA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings, shape: (batch_size, L, 1024)
          - raw_global: raw text by averaging words, shape: (batch_size, 1024)
    Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(TextSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local words and raw global text
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, L)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final text, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        # new_global = l2norm(new_global, dim=-1)
        # if torch.isnan(new_global).any():
        #     print("Tensor contains nan values:", new_global)
        return new_global


class EncoderSimilarity(nn.Module):
    """
    Compute the image-text similarity by SGR, SAF, AVE
    Args: - img_emb: local region embeddings, shape: (batch_size, 36, 1024)
          - cap_emb: local word embeddings, shape: (batch_size, L, 1024)
    Returns:
        - sim_all: final image-text similarities, shape: (batch_size, batch_size).
    """

    def __init__(self, embed_size, sim_dim, module_name, sgr_step):
        super(EncoderSimilarity, self).__init__()
        self.module_name = module_name
        self.sgr_step = sgr_step
        # v_global_w是VisualSA的实例
        # self.v_global_w = VisualSA(embed_size, 0.4, 36)  # 0.4是dropout rate

        self.t_global_w = TextSA(embed_size, 0.4)  # default=1024
        # sim_tranloc_w是一个线性层
        # embed_size=1024 sim_dim=256
        self.sim_tranloc_w = nn.Linear(embed_size, sim_dim)
        self.sim_tranglo_w = nn.Linear(embed_size, sim_dim)
        # sim_eval_w用来计算最终的相似度
        self.sim_eval_w = nn.Linear(sim_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.SAF_module = AttentionFiltration(sim_dim)
        self.SGR_module = nn.ModuleList(
            [GraphReasoning(sim_dim) for i in range(self.sgr_step)])
        self.init_weights()

    def forward(self, img_emb, code_global_emb, cap_emb, cap_lens):  # 代码局部特征 代码全局特征 文本剧本特征
        sim_all = []  # 存放最终的相似度
        n_image = img_emb.size(0)  # n_image是batch_size
        n_caption = cap_emb.size(0)  # n_caption是batch_size

        # # get enhanced global images by self-attention
        # # img_ave是每个图像的平均值 shape: (batch_size, 1024)
        # # img_glo是每个图像的全局特征 shape: (batch_size, 1024)
        # img_glo = self.v_global_w(img_emb, img_ave)
        img_glo = code_global_emb

        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]  # n_word是第i个句子的长度
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)  # cap_i是第i个句子的特征
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            # get enhanced global i-th text by self-attention
            cap_ave_i = th.mean(cap_i, 1)  # cap_ave_i是第i个句子的平均值
            cap_glo_i = self.t_global_w(
                cap_i, cap_ave_i)  # cap_glo_i是第i个句子的全局特征

            # local-global alignment construction
            Context_img = SCAN_attention(
                cap_i_expand, img_emb, smooth=9.0)  # 第i个句子的特征 和代码的局部特征 然后每个句子和
            # sim_loc是第i个句子的局部特征 torch.pow是对应元素相乘 shape: (batch_size, 36, 1024) torch.sub是对应元素相减
            sim_loc = th.pow(th.sub(Context_img, cap_i_expand), 2)  # 归一化

            sim_loc = self.sim_tranloc_w(sim_loc)
            # sim_glo = torch.pow(
            #     torch.sub(self.dim_match_layer(img_glo), cap_glo_i), 2)
            sim_glo = th.pow(
                th.sub(img_glo, cap_glo_i), 2)
            sim_glo = self.sim_tranglo_w(sim_glo)  # shape (batch_size, 256)
            # sim_glo = l2norm(self.sim_tranglo_w(sim_glo), dim=-1)
            # concat the global and local alignments shape: (batch_size, 2, 1024)
            sim_emb = th.cat(
                [sim_glo.unsqueeze(1), sim_loc], 1)  # 将局部特征和全局特征拼接起来
            # compute the final similarity vector
            if self.module_name == 'SGR':
                for module in self.SGR_module:
                    sim_emb = module(sim_emb)
                sim_vec = sim_emb[:, 0, :]
            else:
                sim_vec = self.SAF_module(sim_emb)  # 定位
            # compute the final similarity score
            # sim_i是第i个句子的相似度 sim_vecshape: (batch_size, 256)
            # sim_i shape (batch_size, 1)
            sim_i = self.sigmoid(self.sim_eval_w(sim_vec))
            sim_all.append(sim_i)

        # (n_image, n_caption)
        sim_all = th.cat(sim_all, 1)  # 将所有句子的相似度拼接起来
        # print('sim_all: ', sim_all)
        return sim_all  # shape: (batch_size, batch_size)

    def init_weights(self):
        for m in self.children():  # 遍历所有的子模块
            if isinstance(m, nn.Linear):  # 如果是线性层
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):  # 如果是BN层
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class AttentionFiltration(nn.Module):
    """
    Perform the similarity Attention Filtration with a gate-based attention
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_saf: aggregated alignment after attention filtration, shape: (batch_size, 256)
    """

    def __init__(self, sim_dim=128):
        super(AttentionFiltration, self).__init__()

        self.attn_sim_w = nn.Linear(sim_dim, 1)
        self.bn = nn.BatchNorm1d(1)

        self.init_weights()

    def forward(self, sim_emb):
        # print("sim_emb", sim_emb[0])  # [128, 21, 128]
        sim_attn = l1norm(th.relu(
            self.bn(self.attn_sim_w(sim_emb).permute(0, 2, 1))), dim=-1)
        sim_saf = th.matmul(sim_attn, sim_emb)
        # sim_saf = l2norm(sim_saf.squeeze(1), dim=-1)
        sim_saf = sim_saf.squeeze(1)
        # if torch.isnan(sim_saf).any():
        #     print("Tensor contains nan values:", sim_saf)
        return sim_saf

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class GraphReasoning(nn.Module):
    """
    Perform the similarity graph reasoning with a full-connected graph
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_sgr: reasoned graph nodes after several steps, shape: (batch_size, L+1, 256)
    """

    def __init__(self, sim_dim):
        super(GraphReasoning, self).__init__()

        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU()

        # self.init_weights()

    def forward(self, sim_emb):
        sim_query = self.graph_query_w(sim_emb)
        sim_key = self.graph_key_w(sim_emb)
        sim_edge = th.softmax(
            th.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)
        sim_sgr = th.bmm(sim_edge, sim_emb)
        sim_sgr = self.relu(self.sim_graph_w(sim_sgr))
        return sim_sgr

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def SCAN_attention(query, context, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d) n_context是batch_size， queryL是句子长度  d是特征维度 文本
    context: (n_context, sourceL, d) n_context是batch_size， sourceL是句子长度  d是特征维度 图像
    """
    # --> (batch, d, queryL)
    queryT = th.transpose(query, 1, 2)  # queryT是query的转置 句子上下文

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = th.bmm(context, queryT)  # torch.bmm是矩阵乘法 attn是注意力矩阵

    attn = nn.LeakyReLU(0.1)(attn)  # LeakyReLU是激活函数
    # attn = l2norm(attn, 2)  # l2norm是归一化

    # --> (batch, queryL, sourceL)
    attn = th.transpose(attn, 1, 2).contiguous()  # attn是注意力矩阵的转置
    # --> (batch, queryL, sourceL
    attn = F.softmax(attn * smooth, dim=2)  # F.softmax是softmax函数

    # --> (batch, sourceL, queryL)
    attnT = th.transpose(attn, 1, 2).contiguous()  # attnT是注意力矩阵的转置

    # --> (batch, d, sourceL)
    contextT = th.transpose(context, 1, 2)  # contextT是context的转置 图像上下文
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = th.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = th.transpose(weightedContext, 1, 2)
    # weightedContext = l2norm(weightedContext, dim=-1)

    return weightedContext  # weightedContext代表的是这个句子的上下文特征与图像的上下文特征的加权和


class RealLengthExtractor:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def extract_real_lengths(self, input_tensor):
        pad_mask = (input_tensor != self.pad_token_id)
        rel_lengths = pad_mask.sum(dim=1).int()
        return rel_lengths



class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)  # diagonal是scores对角线上的元素
        # diagonal.expand_as(scores)是将diagonal扩展成和scores一样的形状
        d1 = diagonal.expand_as(scores)
        # diagonal.t().expand_as(scores)是将diagonal转置后扩展成和scores一样的形状
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = th.eye(scores.size(0)) > .5
        if th.cuda.is_available():
            I = mask.to(scores.device)
        I = mask.to(scores.device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()









        #num_graphs, max_num_nodes, _ = node_feat.shape
        #deg_emb = self.degree_encoder(th.stack((in_degree, out_degree)))

        # node feature + degree encoding as input
        #node_feat = self.atom_encoder(node_feat.int()).sum(dim=-2)
        #node_feat = node_feat + deg_emb
        #graph_token_feat = self.graph_token.weight.unsqueeze(0).repeat(
        #    num_graphs, 1, 1
        #)
        #x = th.cat([graph_token_feat, node_feat], dim=1)

        # spatial encoding and path encoding serve as attention bias
        # Since the virtual node comes first, the spatial encodings between it
        # and other nodes will fill the 1st row and 1st column (omit num_graphs
        # and num_heads dimensions) of attn_bias matrix by broadcasting.
        #attn_bias[:, 1:, 0, :] = attn_bias[:, 1:, 0, :] + t
        #attn_bias[:, 0, :, :] = attn_bias[:, 0, :, :] + t
        #x = self.token_embeddings(inputbatch)
        #print(x.shape)