import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import Recurrent_TriangularCausalMask


class Recurrent_FullAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False, n_feature=7):
        super(Recurrent_FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.n_feature = n_feature

    # attention weight size 96 * 96 * 7 * 7

    def forward(self, queries, keys, values, attn_mask, visualize=False):

        # q, k, v 線形のNNを通ってくるが最初は同じ
        B, L, R = queries.shape
        B, S, P = values.shape
        # 32 * 96 * 70

        # 32 * 96 * 10 * 7
        queries = queries.view(B, L, -1, self.n_feature)
        values = values.view(B, L, -1, self.n_feature)
        keys = keys.view(B, L, -1, self.n_feature)

        attention_map = []

        for index_target_time in range(L):
            # self.n_feature 個分のattention mapを作成する
            # 必要なq,kを抽出
            extracted_query = queries[:, index_target_time, :, :]
            # extracted_query = extracted_query.view(B,-1,)
            # print("extracted_query.shape : "+str(extracted_query.shape))

            extracted_key = keys[:, 0:index_target_time+1, :, :]
            # print("extracted_key.shape : "+str(extracted_key.shape))

            extracted_values = values[:, 0:index_target_time+1, :, :]
            # print("extracted_values.shape : "+str(extracted_values.shape))

            # attention map を作成
            scores = torch.einsum("ber,bsep->bspr",
                                  extracted_query, extracted_key)
            # mask するのは
            if self.mask_flag:
                if attn_mask is None:
                    attn_mask = Recurrent_TriangularCausalMask(
                        B, index_target_time+1, self.n_feature, self.n_feature, device=queries.device)

                scores.masked_fill_(attn_mask.mask, -np.inf)

            scores = scores.reshape(B, -1, self.n_feature)
            scores = torch.softmax(scores, dim=-2)
            scores = scores.reshape(B, index_target_time+1, self.n_feature, -1)

            attention_map.append(scores)

            # print("extracted_values.shape : "+str(extracted_values.shape))
            # print("scores.shape : "+str(scores.shape))

            # valueに適用
            V = torch.einsum("bsep,bspr->ber", extracted_values, scores)
            V = V.reshape(B, 1,  -1, self.n_feature)
            # print("V.shape : "+str(V.shape))
            # valueを更新
            values = torch.cat((
                values[:, :index_target_time, :, :], V, values[:, index_target_time+1:, :, :]), 1)

            keys = torch.cat((
                values[:, :index_target_time, :, :], V, values[:, index_target_time+1:, :, :]), 1)
            # values[:, index_target_time+1, :, :] = V
            # print("values.shape : "+str(values.shape))

        if visualize:
            # attention_map = attention_map.to('cpu').detach().numpy().copy()
            np.save("./results/attention/recurrent_attention_map.npy",
                    attention_map)
        if self.output_attention:
            return (values.contiguous(), None)
        else:
            return (values.contiguous(), None)


class Recurrent_AttentionLayer(nn.Module):
    def __init__(self, attention, d_feature=10, n_feature=7,
                 d_keys=None, d_values=None, mix=False):
        super(Recurrent_AttentionLayer, self).__init__()

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


class Recurrent_AttentionLayer_embed_dimension(nn.Module):
    def __init__(self, attention, d_feature=10, n_feature=7,
                 d_keys=None, d_values=None, mix=False):
        super(Recurrent_AttentionLayer, self).__init__()

        self.seq_len = 48

        # d_keys = d_keys or (d_model//n_heads)
        # d_values = d_values or (d_model//n_heads)
        self.d_model = d_feature * n_feature
        self.linear_size = self.d_model*self.seq_len

        self.inner_attention = attention
        # query_projectionがAttentionを計算するための学習対象になる
        self.query_projection = nn.Linear(self.d_model, self.d_model)
        self.key_projection = nn.Linear(self.d_model, self.d_model)
        self.value_projection = nn.Linear(self.d_model, self.d_model)
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

        # queries = queries.reshape(B, -1)
        # keys = keys.reshape(B, -1)
        # values = values.reshape(B, -1)

        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask, visualize=visualize
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()

        # out = out.view(B, L, -1)

        return self.out_projection(out), attn
