import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class CAT_PositionalEmbedding(nn.Module):
    def __init__(self, d_feature=10, max_len=5000):  # この5000は96まででいい
        super(CAT_PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        # zeros 5000 * 10
        pe = torch.zeros(max_len, d_feature).float()
        # 微分の対象とはしない
        pe.require_grad = False

        # [0,1]
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_feature, 2).float()
                    * -(math.log(10000.0) / d_feature)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # torch.Size([1, 5000, 10])
        self.register_buffer('pe', pe)

    def forward(self, x):
        # # torch.Size([1, 96, 10]) -> torch.Size([10, 1, 96])
        tmp_pe = self.pe[:, :x.size(1)].permute(2, 0, 1)
        # print("CAT_PositionalEmbedding forward : " + str(tmp_pe.size()))
        return tmp_pe


class CAT_TokenEmbedding(nn.Module):
    def __init__(self, c_in=1, d_feature=10):
        # 1 チャネルを10チャネルに拡張する

        # print("c_in : "+str(c_in))
        # print("d_feature : "+str(d_feature))

        super(CAT_TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # c_inがなぜか7になっている
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_feature,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        # in_channel 7 out_channel 512
        for m in self.modules():  # このクラスで宣言されたすべての層を見つける
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')  # 重みの初期化

    def forward(self, x: torch.Tensor):  # どう考えてもこのxは二次元な気がする いや三次元 バッチ数があるので二次元から三次元になる
        # print("CAT_TokenEmbedding forward : " + str(x.size()))
        # torch.Size([32, 96]) -> torch.Size([32, 1, 96]) unsqueeze
        x = x.unsqueeze(1)
        # print("CAT_TokenEmbedding unsqueeze : " + str(x.size()))
        x = x.transpose(0, 2)
        # print("CAT_TokenEmbedding transpose : " + str(x.size()))
        x = self.tokenConv(x).permute(1, 2, 0)
        # torch.Size([32, 1, 96]) ->  torch.Size([32, 10, 96]) tokenConv
        # torch.Size([32, 10, 96])-> torch.Size([10, 32, 96]) permute
        # print("CAT_TokenEmbedding permute : " + str(x.size()))
        return x


class CAT_FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(CAT_FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class CAT_TemporalEmbedding(nn.Module):
    # ある時点一点のみを見た時のその埋め込み表現を作成する
    def __init__(self, d_feature=10, embed_type='fixed', freq='h'):
        super(CAT_TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        # 10次元を持つ固有のベクトルに変換する.
        Embed = CAT_FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_feature)
        self.hour_embed = Embed(hour_size, d_feature)
        self.weekday_embed = Embed(weekday_size, d_feature)
        self.day_embed = Embed(day_size, d_feature)
        self.month_embed = Embed(month_size, d_feature)

    def forward(self, x):
        # print("CAT_TemporalEmbedding input : " +
        #       str(x.size()))

        # このxはx_mark
        # temporalenb はmarkを使う
        x = x.long()

        # ある時刻に対しての埋め込み表現は一次元の配列で
        # 月, 日付, 時間, 分の順で並んでいる
        # 以下のスライシングの最後の数字はそれを示す
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        # torch.Size([32, 96, 10])
        # print("hour_x : " + str(hour_x.size()))
        # print("weekday_x : " + str(weekday_x.size()))
        # print("day_x : " + str(day_x.size()))
        # print("month_x : " + str(month_x.size()))
        # print("minute_x : " + str(minute_x.size()))

        temporal_embed = hour_x + weekday_x + day_x + month_x + minute_x

        temporal_embed = temporal_embed.permute(2, 0, 1)
        # print("CAT_TemporalEmbedding Output : " + str(temporal_embed.size()))

        return temporal_embed


class CAT_TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(CAT_TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):

        return self.embed(x)


class CAT_DataEmbedding(nn.Module):
    def __init__(self, c_in=1, d_feature=10, n_feature=7, embed_type='fixed', freq='h', dropout=0.1):
        super(CAT_DataEmbedding, self).__init__()
        self.n_feature = 7

        self.value_embedding = CAT_TokenEmbedding(
            c_in=c_in, d_feature=d_feature)  # CNNにより次元方向に拡張
        # 96のなかでどこに位置しているかを埋め込み表現
        self.position_embedding = CAT_PositionalEmbedding(d_feature=d_feature)
        # ある一時点を見た時のその時刻のみの埋め込み表現
        self.temporal_embedding = CAT_TemporalEmbedding(
            d_feature=d_feature, embed_type=embed_type, freq=freq)
        # if embed_type != 'timeF' else CAT_TimeFeatureEmbedding(
        #     d_model=d_feature, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def _embed(self, x, x_mark):
        # print("_embed : " + str(x.size()))
        # torch.Size([32, 96])
        # value_embedding -> torch.Size([10, 32, 96])
        # position_embedding -> torch.Size([10, 1, 96]) この1は32に自動で合わせられる
        # temporal_embedding -> torch.Size([10, 32, 96])
        embed_one_feature = self.value_embedding(
            x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        # torch.Size([10, 32, 96])
        return embed_one_feature

    def forward(self, x: torch.Tensor, x_mark):
        # n_feature数分繰り返して結合する

        # torch.Size([32, 96, 7]) -> torch.Size([7, 32, 96])
        x = x.permute(2, 0, 1)

        # すべての変数について奥行きを10次元に拡張
        output = torch.cat([self._embed(xi.clone(), x_mark) for xi in x])

        # torch.Size([70, 96, 32]) -> torch.Size([32, 96, 70])
        output = output.permute(1, 2, 0)

        # print("forward transpose in CAT_DataEmbedding :  " + str(output.size()))

        return self.dropout(output)
