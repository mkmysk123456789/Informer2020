import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()  # インスタンス時に呼ばれる 実態を作成したらデータの読み込みが始まる

    def __read_data__(self):
        # __get_item__でデータを取得しやすくするための準備
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))  # ここでは二次元?? というか次元数がない
        # 12 month 30 day 24 hour
        border1s = [0, 12*30*24 - self.seq_len,
                    12*30*24+4*30*24 - self.seq_len]
        # train の場合は0から一年
        # val : 1年4ヶ月の96時点
        # test : 一年4ヶ月から1年八ヶ月
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]  # 複数形
        border2 = border2s[self.set_type]

        # 説明変数の数 multi or single 多変量か単変量か?
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # 時刻の列いがい
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:  # default True 正規化をするかしないか
            train_data = df_data[border1s[0]:border2s[0]]  # trainなら0から2年
            self.scaler.fit(train_data.values)  # データの標準化 標準かをしないほうがいいのか まだしない
            # 正規化の実行 訓練データの平均と分散で正規化する 答えをカンニングしないということ?
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values  # データの取得 nd array

        # これは画像であるPIL image または ndarrayのdata「Height×Width×Channel」を
        # Tensor型のdata「Channel×Height×Width」に変換するというもので,
        # transという変数がその機能を持つことを意味する.
        # なぜChannelの順が入れ替わっているかというと,機械学習をしていく上でChannelが最初のほうが
        # 都合が良いからだと思ってもらって良い.

        # 今の状態でchannelって何??

        df_stamp = df_raw[['date']][border1:border2]  # 必要な行数の時刻を取得
        df_stamp['date'] = pd.to_datetime(df_stamp.date)  # 辞書で呼び出せるように??
        # datatime型がnp.ndarray方になる
        # 二次元 時間方向 × その時刻の表現の埋め込み表現
        # ある時刻に対してその位置埋め込み表現の長さはバラバラ
        # あるデータに対しては同じ
        data_stamp = time_features(
            df_stamp, timeenc=self.timeenc, freq=self.freq)

        # data_xとdata_yで何が違う??

        self.data_x = data[border1:border2]  # データ 二次元
        if self.inverse:  # default false
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
            # transformしていた :
            # していなかった : values[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):  # 自動で呼ばれる 抽象関数の実装?? pythonの機能
        # indexは for in で呼び出される時のindex つまり 0, 1, 2, 3, ...
        s_begin = index  # indexはstart位置
        s_end = s_begin + self.seq_len  # 16日ぶん?? 24*4*4
        r_begin = s_end - self.label_len  # 4日ぶん引く?? 24*4
        r_end = r_begin + self.label_len + self.pred_len  # = s_end + pred_len
        # sに関して
        # indexを基準に時間軸に進める方向に対してseq_len
        # rに関して
        # s_endを基準にして 右にlabel_len 左にlabel_len + pred_len

        seq_x = self.data_x[s_begin:s_end]  # 何時限? 二次元
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]  # markは時刻のマークという意味??
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # x とyがfor in でその後どういう使われ方をするのかに注目
        return seq_x, seq_y, seq_x_mark, seq_y_mark  # for in で呼び出せる

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        #  15分ごとだから4
        border1s = [0, 12*30*24*4 - self.seq_len,
                    12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(
            df_stamp, timeenc=self.timeenc, freq=self.freq)

        # xはエンコーダの入力, yはデコーダの入力 yはtrain時に色々加工される
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4  # この4はどこからきてるの??
            # ラベルをつけるのは1日づつってこと??
            self.label_len = 24*4  # この4はどこからきてるの??
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()  # これは継承もとにない関数

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            print(cols)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len,
                    len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(
            df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_StepCount(Dataset):
    # freq : つまりデータ取得の最小の時間の幅
    # features : 単回帰? 重回帰??
    # size : [どんな時であっても常に同じのモデルに入力される長さ,
    # 予測をするときの正解ラベルの幅 つまり96のうち,
    # 48はすでに答えがわかっている状態でデコーダに入力される,
    # 予測する長さ]
    def __init__(self, root_path='./data/StepCount/', flag='train', size=[96, 48, 24],
                 features='S', data_path='iPhone_StepCount_from_2017_to_2021_6_18.csv',
                 target='StepCount', scale=True, inverse=False, timeenc=0, freq='H', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4  # この4はどこからきてるの??
            # ラベルをつけるのは1日づつってこと??
            self.label_len = 24*4  # この4はどこからきてるの??
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()  # これは継承もとにない関数

    def __read_data__(self):
        self.scaler = StandardScaler()  # 標準化のため
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);
        # データの列に対して並び替えをする IndexはDatatime型にしない 列の'date'がindexになる
        # つまり自分のstepCountを使いたいなら'date'がindexになる
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test  # つまり 1 - 0.7 - 0.2

        # 複数形のs
        # [0 訓練データまでのindex, validation 検証用データまでのindex, テストデータまでのindex]
        border1s = [0, num_train-self.seq_len,
                    len(df_raw)-num_test-self.seq_len]

        border2s = [num_train, num_train+num_vali, len(df_raw)]
        # 取り出したいデータの種類に合わせて
        border1 = border1s[self.set_type]  # どのindexから
        border2 = border2s[self.set_type]  # どのindexまでを取り出すのか

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:  # 標準化するかどうか
            train_data = df_data[border1s[0]:border2s[0]]
            # axis = 0 で平均, 分散を計算
            self.scaler.fit(train_data.values)
            # 平均, 分散により標準化, dfのtransformではない
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        # dataframeからはドットで取り出すこともできる
        # indexはintで持っておいて, date列名にdatatime型で持っておく
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # 最終的にmarkとして扱われる
        data_stamp = time_features(
            df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # markはtemporalenbedingに使う

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='M', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]  # default 96
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        # 予測では最後の値を使う
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(
            tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        # priods 個数指定

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(
            df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):  # これの理解が一番重要??
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
