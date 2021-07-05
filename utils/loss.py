
import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 初期化処理
        # self.param = ...

    def _attention_mask_loss(self, outputs: Tensor, outputs_masked_attention: Tensor) -> Tensor:
        loss_normal = nn.MSELoss(outputs, outputs_masked_attention)
        return 1/loss_normal

    def forward(self, outputs: Tensor, targets: Tensor, outputs_masked_attention: Tensor) -> Tensor:
        '''
        outputs: 予測結果(ネットワークの出力)
        '''
        # 損失の計算
        loss_normal = nn.MSELoss(outputs, targets)
        loss_attention_mask = self._attention_mask_loss(
            outputs, outputs_masked_attention)
        return loss_normal+loss_attention_mask
