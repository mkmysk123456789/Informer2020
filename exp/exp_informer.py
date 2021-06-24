from tqdm import tqdm
from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_StepCount
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)  # モデルの作成, cpu or gpu とか

    def _build_model(self):
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack,
        }
        if self.args.model == 'informer' or self.args.model == 'informerstack':  # このモデルの違いは?
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers  # elayerかslayer
            # エンコーダのレイヤーの数??
            # モデルの設定について保存
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,  # self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model  # toをつけたらtrainが呼び出せる

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,  # 別の場所の変電所のデータ
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
            'StepCount': Dataset_StepCount
        }
        Data = data_dict[self.args.data]  # どのデータを使うか　実態はまだ存在しない
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1  # どういう意味??
            freq = args.detail_freq
            Data = Dataset_Pred  # 予測するならdatasetは予測専用のものに変更する
        else:  # train?? val ??
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        # どのデータを使うか, pathは??
        # DataはDatasetを継承している
        # なので__read_data__
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len,
                  args.pred_len],  # default pred_len 24
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        # batchサイズで扱いやすくなる for in でバッチサイズごとに呼び出せる
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        # data_setは実際のデータ
        # data_loderは
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(vali_loader)):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        # データを取得, pytorchのライブラリを活用
        # data_set, data_loader
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')  # val??
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()  # lossの計算方法

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):  # epoch 初期値は 6
            iter_count = 0
            train_loss = []

            self.model.train()  # 1. modelのtrainを呼び出す
            epoch_time = time.time()
            # データローダをfor inで回すことによって扱いやすくなる
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(train_loader)):
                print("Shape of batch_x")
                print(batch_x.shape)
                iter_count += 1

                model_optim.zero_grad()  # 勾配の初期化
                # 学習時は　model.eval()を呼ばない
                # ここからが本質 xとyが何者なのか
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)  # 現在の出力と正しい値
                loss = criterion(pred, true)  # 誤差計算
                train_loss.append(loss.item())

                if (i+1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed * \
                        ((self.args.train_epochs - epoch)*train_steps - i)
                    print(
                        '\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()  # 誤差逆伝搬
                    model_optim.step()  # 更新

            # loss のデータをsaveしたい

            print("Epoch: {} cost time: {}".format(
                epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)

        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))  # いつセーブした?

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        # print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shsape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy',
                np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            # checkpoint default "checkpoints"
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        # Batch normは、学習の際はバッチ間の平均や分散を計算しています。
        # 推論するときは、平均/分散の値が正規化のために使われます。
        # まとめると、eval()はdropoutやbatch normの on/offの切替です。

        preds = []
        trues = []

        # i = 0 しか実行しない bach = 1??
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            # どこの予測??
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

            # print("Shape of pred on prediction:{}".format(true.shape))
            # print("Shape of true on prediction:{}".format(true.shape))

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        trues = np.array(trues)
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path+'real_prediction.npy', preds)
        np.save(folder_path+'real_prediction_trues.npy', trues)

        return

    # 一回のbatchに対してのモデル全体を通して出力を計算, 正解の値も返す
    # train, val, pred すべてこれを使う
    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # print("Shape of batch_x on top model:{}".format(batch_x.shape))
        # print("Shape of batch_y on top model:{}".format(batch_y.shape))

        # xがエンコーダ, yがデコーダのインプット
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input 予測したいところを0埋め込み
        # 具体的にどこを埋め込み?
        if self.args.padding == 0:
            # Tensor型のdata「Channel×Height×Width」に変換するというもので,
            # 7 * 時間方向 *
            # batch_yはつまり予測したい値, ０に置き換える配列の大きさ??
            dec_inp = torch.zeros(
                [batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones(
                [batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        # 列方向につなげる
        # label len は 48 すでにわかっていることを前提で予測をする長さ
        # label_lenは正解ラベル, pred_lenは予測する長さで0とおく

        dec_inp = torch.cat(
            [batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # print("Shape of dec_inp on top model:{}".format(dec_inp.shape))

        # encoder - decoder　modelの出力, エンコーダとデコーダを通ってきた結果を出力
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    # bacth xは三次元?
                    outputs = self.model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark,
                                     dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark,
                                     dec_inp, batch_y_mark)
        if self.args.inverse:  # default false
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        # outputの次元数
        # print("Shape of output on top model:{}".format(outputs.shape))

        return outputs, batch_y  # batch_y 正解
