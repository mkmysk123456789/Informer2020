import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask, CAT_TriangularCausalMask


class CAT_FullAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(CAT_FullAttention, self).__init__()
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

        scores = torch.einsum("blep,bser->blspr", queries, keys)
        # print(str(visualize))

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = CAT_TriangularCausalMask(
                    B, L, 7, 7, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        if visualize:
            attention_weight = scores.to('cpu').detach().numpy().copy()
            np.save("./results/attention/attention_weight.npy",
                    attention_weight)

        scores = scores.reshape(B, L, -1, 7)
        scores = torch.softmax(scores, dim=-2)
        scores = scores.reshape(B, 96, 96, 7, -1)

        A = self.dropout(scores)

        # V = torch.einsum("bser,blspr->blep", values, A)
        V = torch.einsum("bsep,blspr->bler", values, A)
        V = V.reshape(B, L, 70)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

    # # attention weight size 96 * 96 * 70 * 70
    # def forward(self, queries, keys, values, attn_mask, visualize=False):

    #     # q, k, v 線形のNNを通ってくるが最初は同じ
    #     B, L, R = queries.shape
    #     B, S, P = values.shape
    #     # scale = self.scale or 1./sqrt(E)

    #     scores = torch.einsum("blr,bsp->blspr", queries, keys)
    #     # print(str(visualize))
    #     # これを可視化するにはどうしたらいいか
    #     if visualize:
    #         attention_weight = scores.to('cpu').detach().numpy().copy()
    #         np.save("./results/attention/attention_weight.npy",
    #                 attention_weight)

    #     # if self.mask_flag:
    #     #     if attn_mask is None:
    #     #         attn_mask = TriangularCausalMask(B, L, device=queries.device)

    #     #     # 動的ネットワーク?? 流れてくるデータに応じて構造が変わる??
    #     #     scores.masked_fill_(attn_mask.mask, -np.inf)

    #     scores = scores.view(B, L, -1, R)
    #     scores = torch.softmax(scores, dim=-2)
    #     scores = scores.view(B, L, S, P, -1)

    #     A = self.dropout(scores)

    #     V = torch.einsum("bsp,blspr->blr", values, A)

    #     if self.output_attention:
    #         return (V.contiguous(), A)
    #     else:
    #         return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert(L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask, visualize=False):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
            np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class CAT_AttentionLayer(nn.Module):
    def __init__(self, attention, d_feature=10, n_feature=7,
                 d_keys=None, d_values=None, mix=False):
        super(CAT_AttentionLayer, self).__init__()

        self.seq_len = 96

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
