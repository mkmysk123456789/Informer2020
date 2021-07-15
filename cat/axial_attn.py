import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import Axial_TriangularCausalMask


class Axial_Attention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(Axial_Attention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    # attention weight size 96 * 96 * 7 * 7

    def forward(self, queries, keys, values, attn_mask, visualize=False):

        # q, k, v 線形のNNを通ってくるが最初は同じ
        B, L, R = queries.shape
        B, S, P = values.shape
        # scale = self.scale or 1./sqrt(E)

        # 32 * 96 * 10 * 7
        queries = queries.view(B, L, -1, 7)
        values = values.view(B, L, -1, 7)
        keys = keys.view(B, L, -1, 7)

        B, L, E, R = queries.shape
        B, S, E, P = values.shape

        # 時間方向のattention
        scores_time = torch.einsum("bler,bsep->bls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = Axial_TriangularCausalMask(
                    B, L, L, device=queries.device)

            scores_time.masked_fill_(attn_mask.mask, -np.inf)

        if visualize:
            attention_map_time = scores_time.to('cpu').detach().numpy().copy()
            np.save("./results/attention/attention_map_time.npy",
                    attention_map_time)

        scores_time = torch.softmax(scores_time, dim=-2)

        A_time = self.dropout(scores_time)

        # attention map 適用
        # V = torch.einsum("bsep,bls->bler", values, A_time)
        # V = V.reshape(B, L, -1)

        # 特徴量方向
        scores_feature = torch.einsum("bler,bsep->brp", queries, keys)

        if self.mask_flag:
            attn_mask = Axial_TriangularCausalMask(
                B, R, R, device=queries.device)

            scores_feature.masked_fill_(attn_mask.mask, -np.inf)

        if visualize:
            attention_map_feature = scores_feature.to(
                'cpu').detach().numpy().copy()
            np.save("./results/attention/attention_map_feature.npy",
                    attention_map_feature)

        scores_feature = torch.softmax(scores_feature, dim=-2)

        A_feature = self.dropout(scores_feature)

        # 二つのattention mapから一つのattention mapを作る
        A_time_feature = torch.einsum("bls,brp->blr", A_time, A_feature)
        # 二次元でsoftmax
        A_time_feature = A_time_feature.view(B, -1)
        A_time_feature = torch.softmax(A_time_feature, dim=-1)
        A_time_feature = A_time_feature.view(B, L, -1)

        # attention map 適用
        V = torch.einsum("bsep,blr->bler", values, A_time_feature)
        V = V.reshape(B, L, -1)

        if self.output_attention:
            return (V.contiguous(), A_time)
        else:
            return (V.contiguous(), None)


class Axial_AttentionLayer(nn.Module):
    def __init__(self, attention, d_feature=10, n_feature=7,
                 d_keys=None, d_values=None, mix=False):
        super(Axial_AttentionLayer, self).__init__()

        self.seq_len = 48

        # d_keys = d_keys or (d_model//n_heads)
        # d_values = d_values or (d_model//n_heads)
        self.d_model = d_feature * n_feature
        self.linear_size = self.d_model*self.seq_len

        self.inner_attention = attention
        # query_projectionがAttentionを計算するための学習対象になる
        self.query_projection = nn.Linear(self.linear_size, self.linear_size)
        self.key_projection = nn.Linear(self.linear_size, self.linear_size)
        self.value_projection = nn.Linear(self.linear_size, self.linear_size)
        self.out_projection = nn.Linear(self.d_model, self.d_model)
        # self.n_heads = n_heads  # 512
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask, visualize=False):
        # x , cross, cross なのでエンコーダの出力がkeyとqueriesになる

        # print("queries.shape : "+str(queries.shape))
        # print("keys.shape : "+str(keys.shape))

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        # H = self.n_heads

        # query [32,96,70]

        queries = queries.reshape(B, -1)
        keys = keys.reshape(B, -1)
        values = values.reshape(B, -1)

        queries = self.query_projection(queries).view(B, L, self.d_model)
        keys = self.key_projection(keys).view(B, S, self.d_model)
        values = self.value_projection(values).view(B, S, self.d_model)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask, visualize=visualize
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
