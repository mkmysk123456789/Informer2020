import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

from cat.cat_attn import CAT_FullAttention, CAT_AttentionLayer
from cat.cat_encorder import CAT_Encoder, CAT_EncoderLayer
from cat.cat_decorder import CAT_Decoder, CAT_DecoderLayer
from cat.cat_embed import CAT_DataEmbedding


class CAT(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_feature=10, n_feature=7, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='normal', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(CAT, self).__init__()
        self.pred_len = out_len  # 予測する長さ
        self.attn = attn
        self.output_attention = output_attention

        self.d_model = d_feature * n_feature

        # Encoding
        self.enc_embedding = CAT_DataEmbedding(
            1, d_feature, n_feature, embed, freq, dropout)
        self.dec_embedding = CAT_DataEmbedding(
            1, d_feature, n_feature, embed, freq, dropout)
        # Attention
        # Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = CAT_Encoder(
            [
                CAT_EncoderLayer(
                    CAT_AttentionLayer(CAT_FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                       d_feature=d_feature, n_feature=n_feature, mix=False),
                    d_feature=d_feature,
                    n_feature=n_feature,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    self.d_model
                ) for l in range(e_layers-1)
            ] if distil else None,  # default false
            norm_layer=torch.nn.LayerNorm(self.d_model)  # これを変化??
        )
        # Decoder
        self.decoder = CAT_Decoder(
            [
                CAT_DecoderLayer(
                    CAT_AttentionLayer(CAT_FullAttention(True, factor, attention_dropout=dropout, output_attention=False),
                                       d_feature=d_feature, n_feature=n_feature,  mix=mix),
                    CAT_AttentionLayer(CAT_FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                       d_feature=d_feature, n_feature=n_feature, mix=False),
                    d_feature=d_feature,
                    n_feature=n_feature,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(self.d_model, c_out, bias=True)
        # c_out = 7

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)

        # デコーダの入力にエンコーダの出力を入れる
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


# エンコーダがたくさんあるということ??
class CAT_InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=[3, 2, 1], d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(CAT_InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(
            enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(
            dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder

        # [0,1,2,...] you can customize here
        inp_lens = list(range(len(e_layers)))
        encoders = [
            Encoder(
                [
                    # attention + conv
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                       d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # モデルに入力されるbacth_xの次元数を確認 forwardの中のprintは無視される
        # print("Shape of x_enc on top model:{}".format(x_enc.shape))
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # mark?? 埋め込み表現にする
        enc_out, attns = self.encoder(
            enc_out, attn_mask=enc_self_mask)  # エンコーダの計算

        dec_out = self.dec_embedding(x_dec, x_mark_dec)  # デコーダの埋め込み表現
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)  # デコーダの計算
        dec_out = self.projection(dec_out)  # 線形正規化

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]