# Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (AAAI'21 Best Paper)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![cuDNN 7.3.1](https://img.shields.io/badge/cudnn-7.3.1-green.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

This is the origin Pytorch implementation of Informer in the following paper: 
[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436). Special thanks to `Jieqi Peng`@[cookieminions](https://github.com/cookieminions) for building this repo.

:triangular_flag_on_post:**News**(Mar 25, 2021): We update all experiment [results](#resultslink) with hyperparameter settings.

:triangular_flag_on_post:**News**(Feb 22, 2021): We provide [Colab Examples](#colablink) for friendly usage.

:triangular_flag_on_post:**News**(Feb 8, 2021): Our Informer paper has been awarded [AAAI'21 Best Paper](https://www.business.rutgers.edu/news/hui-xiong-and-research-colleagues-receive-aaai-best-paper-award)! We will continue this line of research and update on this repo. Please star this repo and [cite](#citelink) our paper if you find our work is helpful for you.

<p align="center">
<img src=".\img\informer.png" height = "360" alt="" align=center />
<br><br>
<b>Figure 1.</b> The architecture of Informer.
</p>

## ProbSparse Attention
The self-attention scores form a long-tail distribution, where the "active" queries lie in the "head" scores and "lazy" queries lie in the "tail" area. We designed the ProbSparse Attention to select the "active" queries rather than the "lazy" queries. The ProbSparse Attention with Top-u queries forms a sparse Transformer by the probability distribution.
`Why not use Top-u keys?` The self-attention layer's output is the re-represent of input. It is formulated as a weighted combination of values w.r.t. the score of dot-product pairs. The top queries with full keys encourage a complete re-represent of leading components in the input, and it is equivalent to selecting the "head" scores among all the dot-product pairs. If we choose Top-u keys, the full keys just preserve the trivial sum of values within the "long tail" scores but wreck the leading components' re-represent.
<p align="center">
<img src=".\img\probsparse_intro.png" height = "320" alt="" align=center />
<br><br>
<b>Figure 2.</b> The illustration of ProbSparse Attention.
</p>

## Requirements

- Python 3.6
- matplotlib == 3.1.1
- numpy == 1.19.4
- pandas == 0.25.1
- scikit_learn == 0.21.3
- torch == 1.8.0

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data

The ETT dataset used in the paper can be download in the repo [ETDataset](https://github.com/zhouhaoyi/ETDataset).
The required data files should be put into `data/ETT/` folder. A demo slice of the ETT data is illustrated in the following figure. Note that the input of each dataset is zero-mean normalized in this implementation.

<p align="center">
<img src="./img/data.png" height = "168" alt="" align=center />
<br><br>
<b>Figure 3.</b> An example of the ETT data.
</p>


## Usage
<span id="colablink">Colab Examples:</span> We provide google colabs to help reproducing and customing our repo, which includes `experiments(train and test)`, `prediction`, `visualization` and `custom data`.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_X7O2BkFLvqyCdZzDZvV2MB0aAvYALLC)

Commands for training and testing the model with *ProbSparse* self-attention on Dataset ETTh1, ETTh2 and ETTm1 respectively:

```bash
# ETTh1
python3 -u main_informer.py --model informer --data ETTh1 --attn prob --freq h

# ETTh2
python -u main_informer.py --model informer --data ETTh2 --attn prob --freq h

# ETTm1
python -u main_informer.py --model informer --data ETTm1 --attn prob --freq t
```

More parameter information please refer to `main_informer.py`.


## <span id="resultslink">Results</span>

We have updated the experiment results of all methods due to the change in data scaling. We are lucky that Informer gets performance improvement. Thank you @lk1983823 for reminding the data scaling in [issue 41](https://github.com/zhouhaoyi/Informer2020/issues/41).

Besides, the experiment parameters of each data set are formated in the `.sh` files in the directory `./scripts/`. You can refer to these parameters for experiments, and you can also adjust the parameters to obtain better mse and mae results or draw better prediction figures.

<p align="center">
<img src="./img/result_univariate.png" height = "500" alt="" align=center />
<br><br>
<b>Figure 4.</b> Univariate forecasting results.
</p>

<p align="center">
<img src="./img/result_multivariate.png" height = "500" alt="" align=center />
<br><br>
<b>Figure 5.</b> Multivariate forecasting results.
</p>


## FAQ
If you run into a problem like `RuntimeError: The size of tensor a (98) must match the size of tensor b (96) at non-singleton dimension 1`, you can check torch version or modify code about `Conv1d` of `TokenEmbedding` in `models/embed.py` as the way of circular padding mode in Conv1d changed in different torch versions.


## <span id="citelink">Citation</span>
If you find this repository useful in your research, please consider citing the following paper:

```
@inproceedings{haoyietal-informer-2021,
  author    = {Haoyi Zhou and
               Shanghang Zhang and
               Jieqi Peng and
               Shuai Zhang and
               Jianxin Li and
               Hui Xiong and
               Wancai Zhang},
  title     = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  booktitle = {The Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021},
  pages     = {online},
  publisher = {{AAAI} Press},
  year      = {2021},
}
```

## Contact
If you have any questions, feel free to contact Haoyi Zhou through Email (zhouhaoyi1991@gmail.com) or Github issues. Pull requests are highly welcomed!

## Acknowlegements
Thanks for the computing infrastructure provided by Beijing Advanced Innovation Center for Big Data and Brain Computing ([BDBC](http://bdbc.buaa.edu.cn/)).
At the same time, thank you all for your attention to this work!
[![Stargazers repo roster for @zhouhaoyi/Informer2020](https://reporoster.com/stars/zhouhaoyi/Informer2020)](https://github.com/zhouhaoyi/Informer2020/stargazers)



# 再現実装用 実行コマンド集

## StepCount

### 1時間

prob

python3 -u main_informer.py --model informer --data StepCount --attn prob --freq h --data_path iPhone_StepCount_from_2017_to_2021_6_18.csv --root_path ./data/StepCount --target StepCount --enc_in 1 --dec_in 1 --c_out 1 --features S --pred_len 48

full

python3 -u main_informer.py --model informer --data StepCount --attn full --freq h --data_path iPhone_StepCount_from_2017_to_2021_6_18.csv --root_path ./data/StepCount --target StepCount --enc_in 1 --dec_in 1 --c_out 1 --features S --pred_len 48

train False

python3 -u main_informer.py --model CAT --data ETTh1 --attn full --freq h  --enc_in 1 --dec_in 1 --c_out 1  --pred_len 24 --d_feature 10 --n_feature 7 --label_len 72 --batch_size 10 --train_epochs 2 --train False


CAT 

python3 -u main_informer.py --model CAT --data ETTh1 --attn full --freq h  --enc_in 1 --dec_in 1 --c_out 1  --pred_len 24 --d_feature 10 --n_feature 7 --label_len 72

python3 -u main_informer.py --model CAT --data ETTh1 --attn full --freq h  --enc_in 1 --dec_in 1 --c_out 1  --pred_len 24 --d_feature 10 --n_feature 7 --label_len 24 --batch_size 32 --train_epochs 20 --seq_len 48


CAT layer

python3 -u main_informer.py --model CAT --data ETTh1 --attn full --freq h  --enc_in 1 --dec_in 1 --c_out 1  --pred_len 24 --d_feature 10 --n_feature 7 --label_len 24 --batch_size 32 --train_epochs 10 --seq_len 48 --e_layers 2 --d_layers 1 --itr 1




python3 -u main_informer.py --model CAT --data ETTh1 --attn full --freq h  --enc_in 1 --dec_in 1 --c_out 1  --pred_len 24 --d_feature 10 --n_feature 7 --label_len 24 --batch_size 32  --train_epochs 10 --seq_len 48 --e_layers 2 --d_layers 1 --itr 1


d_feature change

python3 -u main_informer.py --model CAT --data ETTh1 --attn full --freq h  --enc_in 1 --dec_in 1 --c_out 1  --pred_len 24 --d_feature 32 --n_feature 7 --label_len 24 --batch_size 32  --train_epochs 10 --seq_len 48 --e_layers 2 --d_layers 1 --itr 1



python3 -u main_informer.py --model CAT --data ETTh1 --attn full --freq h  --enc_in 1 --dec_in 1 --c_out 1  --pred_len 24 --d_feature 48 --n_feature 7 --label_len 24 --batch_size 32  --train_epochs 10 --seq_len 48 --e_layers 2 --d_layers 1 --itr 1


axial test

python3 -u main_informer.py --model CAT --data ETTh1 --attn Axial --freq h  --enc_in 1 --dec_in 1 --c_out 1  --pred_len 24 --d_feature 32 --n_feature 7 --label_len 24 --batch_size 32  --train_epochs 10 --seq_len 48 --e_layers 2 --d_layers 1 --itr 1

Best Result

python3 -u main_informer.py --model CAT --data ETTh1 --attn CAT --freq h  --enc_in 1 --dec_in 1 --c_out 1  --pred_len 24 --d_feature 32 --n_feature 7 --label_len 24 --batch_size 32  --train_epochs 10 --seq_len 48 --e_layers 2 --d_layers 1 --itr 1 --notify True


Recurrent Experiment

python3 -u main_informer.py --model CAT --data ETTh1 --attn Recurrent --freq h  --enc_in 1 --dec_in 1 --c_out 1  --pred_len 24 --d_feature 32 --n_feature 7 --label_len 24 --batch_size 32  --train_epochs 10 --seq_len 48 --e_layers 2 --d_layers 1 --itr 1 --notify True



 python3 -u main_informer.py --model CAT --data ETTh1 --attn Recurrent --freq h  --enc_in 1 --dec_in 1 --c_out 1  --pred_len 24 --d_feature 32 --n_feature 7 --label_len 24 --batch_size 8  --train_epochs 20 --seq_len 48 --e_layers 2 --d_layers 1 --itr 1 --notify True

  python3 -u main_informer.py --model CAT --data ETTh1 --attn Recurrent --freq h  --enc_in 1 --dec_in 1 --c_out 1  --pred_len 24 --d_feature 24 --n_feature 7 --label_len 24 --batch_size 16  --train_epochs 20 --seq_len 48 --e_layers 2 --d_layers 1 --itr 10 --notify True


embed linear dimension small parameter

python3 -u main_informer.py --model CAT --data ETTh1 --attn Recurrent_embed_linear --freq h  --enc_in 1 --dec_in 1 --c_out 1  --pred_len 24 --d_feature 64 --n_feature 7 --label_len 24 --batch_size 32  --train_epochs 20 --seq_len 48 --e_layers 2 --d_layers 1 --itr 10 --notify True