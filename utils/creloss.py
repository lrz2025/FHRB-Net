import torch

def cre_loss(prob):
    """
    计算CRE（累积残余熵）损失。
    参数:
        prob: shape=[N, ...]，全体sigmoid概率输出（可1维/2维/任意shape），建议为[batch, h, w]或[batch*N]
    返回:
        cre: 标量张量，表示不确定性
    """
    # 1. 拉平成一维，去掉NaN
    x = prob.reshape(-1)
    x = x[~torch.isnan(x)]
    N = x.shape[0]

    # 2. 升序排序
    x_sorted, _ = torch.sort(x)

    # 3. 构造ECDF并获得生存函数（右尾概率）
    # 计算每个阶梯的右尾概率（生存函数）：1 - i/N, i=1~N-1
    cd = N
    # diff: A(2:end)-A(1:end-1)
    diffs = x_sorted[1:] - x_sorted[:-1]
    idx = torch.arange(1, cd, device=x.device)   # 1 ~ N-1
    surv = 1.0 - idx / cd                       # (N-1, )

    # 防止log(0)
    surv_clipped = torch.clamp(surv, min=1e-12)
    n = diffs * (surv_clipped * torch.log(surv_clipped))
    cre = -torch.sum(n)
    return cre

# 封装成PyTorch nn.Module损失
import torch.nn as nn
class CRELoss(nn.Module):
    def __init__(self):
        super(CRELoss, self).__init__()
    def forward(self, prob):
        return cre_loss(prob)

# 用法示例
if __name__ == "__main__":
    # 伪造sigmoid输出
    prob = torch.sigmoid(torch.randn(4, 64, 64))
    loss = cre_loss(prob)
    print('CRE loss:', loss.item())
