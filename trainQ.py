# -*- coding: gbk -*-
import os, sys, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
import json
from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 超参（可命令行覆盖）
epoch_n = 5
tau = 2.0
thr = 0.5
beta = 0.5
learnable_beta = False

lambda_q_l1   = 1e-3    # 稀疏化
lambda_q_bin  = 1e-3    # 推 0/1
lambda_q_sup  = 0.5     # 与给定 Q 对齐,BCE，与给定 Q 的弱监督
lambda_q_card = 0.0     # 行基数对齐（可先关），行和接近目标个数

result_dir = None
q_dir = None
log_path = None

def build_and_set_dirs():
    global result_dir, q_dir, log_path
    result_dir = f'result_Q_beta{beta}_l1{lambda_q_l1}_bin{lambda_q_bin}_sup{lambda_q_sup}/'
    os.makedirs(result_dir, exist_ok=True)
    q_dir = os.path.join(result_dir, 'Q')
    os.makedirs(q_dir, exist_ok=True)
    log_path = os.path.join(result_dir, 'model_val.txt')

def parse_args_and_override():
    global epoch_n, tau, thr, beta, learnable_beta
    global lambda_q_l1, lambda_q_bin, lambda_q_sup, lambda_q_card

    ap = argparse.ArgumentParser()
    ap.add_argument('--epoch_n', type=int, default=epoch_n)
    ap.add_argument('--tau', type=float, default=tau)
    ap.add_argument('--thr', type=float, default=thr)
    ap.add_argument('--beta', type=float, default=beta)
    ap.add_argument('--learnable_beta', action='store_true')
    ap.add_argument('--lambda_q_l1', type=float, default=lambda_q_l1)
    ap.add_argument('--lambda_q_bin', type=float, default=lambda_q_bin)
    ap.add_argument('--lambda_q_sup', type=float, default=lambda_q_sup)
    ap.add_argument('--lambda_q_card', type=float, default=lambda_q_card)
    args = ap.parse_args()

    epoch_n = args.epoch_n
    tau = args.tau
    thr = args.thr
    beta = args.beta
    learnable_beta = args.learnable_beta
    lambda_q_l1 = args.lambda_q_l1
    lambda_q_bin = args.lambda_q_bin
    lambda_q_sup = args.lambda_q_sup
    lambda_q_card = args.lambda_q_card


def train_q():
    # 规模
    with open('config.txt') as f:
        f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, f.readline().split(',')))

    # dataloader 要能返回 (stu_id, exer_id, Q_row, label)
    # 其中 Q_row 就是你之前的 kn_emb（题-概念向量）
    train_loader = TrainDataLoader()
    val_loader = ValTestDataLoader('validation')

    net = Net(student_n, exer_n, knowledge_n,
              tau=tau, thr=thr, beta=beta, learnable_beta=learnable_beta).to(device)
    opt = optim.Adam(net.parameters(), lr=2e-3)
    bce_logits = nn.BCEWithLogitsLoss(reduction='mean')
    bce = nn.BCELoss(reduction='mean')

    # 累积全量原始 Q（按题目聚合），在第一个 epoch 结束时保存一次
    Q_given_full = torch.zeros((exer_n, knowledge_n), device=device)

    for ep in range(epoch_n):
        net.train()
        train_loader.reset()
        tot, main_tot, qsup_tot, l1_tot, bin_tot, card_tot = 0,0,0,0,0,0
        steps = 0

        while not train_loader.is_end():
            steps += 1
            stu, exer, q_given, y = train_loader.next_batch()
            stu, exer, q_given, y = stu.to(device), exer.to(device), q_given.to(device), y.to(device)
            y = y.float().view(-1, 1)
            q_given = q_given.float()

            # 累积原始 Q（同题目多次出现取并集/最大值）
            Q_given_full[exer] = torch.maximum(Q_given_full[exer], q_given)

            opt.zero_grad()
            logits = net.forward(stu, exer, q_given=q_given)      # [B,1] logits
            main_loss = bce_logits(logits, y)

            # Q 相关 loss
            q_soft = net.get_q_soft()[exer]                    # [B,K] soft
            # 监督一致
            q_sup = bce(q_soft, q_given)
            # 稀疏
            q_l1 = q_soft.mean(dim=1).mean()
            # 0/1 逼近
            q_bin = (q_soft * (1 - q_soft)).mean()
            # 行基数（可选）：让行和接近给定 Q 的行和（或某个常数 t）
            if lambda_q_card > 0:
                target_card = q_given.sum(dim=1, keepdim=True).clamp(min=1.0)
                card = (q_soft.sum(dim=1, keepdim=True) - target_card).abs().mean()
            else:
                card = q_soft.new_tensor(0.)

            loss = main_loss + \
                   lambda_q_sup * q_sup + \
                   lambda_q_l1  * q_l1 + \
                   lambda_q_bin * q_bin + \
                   lambda_q_card* card

            loss.backward()
            opt.step()

            tot += float(loss); main_tot += float(main_loss)
            qsup_tot += float(q_sup); l1_tot += float(q_l1)
            bin_tot += float(q_bin); card_tot += float(card)

            if steps % 200 == 0:
                print(f'[ep {ep+1} step {steps}] loss={tot/steps:.3f} main={main_tot/steps:.3f} '
                      f'Qsup={qsup_tot/steps:.4f} L1={l1_tot/steps:.5f} bin={bin_tot/steps:.5f}')

        # 验证
        accuracy, rmse, auc = validate(net, val_loader)
        # 写入epoch日志
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(
                'epoch=%d, acc=%.6f, rmse=%.6f, auc=%.6f, '
                'loss=%.6f, main=%.6f, q_sup=%.6f, q_l1=%.6f, q_bin=%.6f, q_card=%.6f\n' %
                (ep+1, accuracy, rmse, auc,
                 tot/max(1,steps), main_tot/max(1,steps),
                 qsup_tot/max(1,steps), l1_tot/max(1,steps),
                 bin_tot/max(1,steps), card_tot/max(1,steps))
            )

        # 导出学到的 Q（硬二值）
        Q_learn = net.export_Q(hard=True).cpu().numpy()   # [E, K], 0/1
        np.save(os.path.join(q_dir, f'Q_learn_epoch{ep+1}.npy'), Q_learn)
        # 导出 soft Q（可解释用）
        Q_soft = net.export_Q(hard=False).cpu().numpy()   # [E, K], 0~1
        np.save(os.path.join(q_dir, f'Q_soft_epoch{ep+1}.npy'), Q_soft)
        # 保存原始 Q（一次即可）
        if ep == 0:
            np.save(os.path.join(q_dir, 'Q_given.npy'), Q_given_full.cpu().numpy())

        # 保存模型
        torch.save(net.state_dict(), os.path.join(result_dir, f'model_epoch{ep+1}'))
        print(f'[epoch {ep+1}] acc={accuracy:.4f} rmse={rmse:.4f} auc={auc:.4f} log->{log_path}')

def validate(net, loader):
    net.eval(); loader.reset()
    preds, labels = [], []
    correct, total = 0, 0
    with torch.no_grad():
        while not loader.is_end():
            stu, exer, q_given, y = loader.next_batch()
            stu, exer, q_given, y = stu.to(device), exer.to(device), q_given.to(device), y.to(device)
            logits = net.forward(stu, exer, q_given=q_given).view(-1)
            p = torch.sigmoid(logits)

            preds += p.cpu().tolist()
            labels += y.cpu().tolist()
            pred_bin = (p > 0.5).long()   # 等价于 logits > 0
            correct += (pred_bin == y).sum().item()
            total   += len(y)

    preds = np.array(preds); labels = np.array(labels)
    acc  = correct / max(1,total)
    rmse = np.sqrt(np.mean((labels - preds) ** 2))
    try:
        auc  = roc_auc_score(labels, preds)
    except ValueError:
        auc = float('nan')
    return acc, rmse, auc

if __name__ == '__main__':
    # 解析命令行并覆盖默认超参
    parse_args_and_override()
    # 按覆盖后的权重构建结果目录
    build_and_set_dirs()
    train_q()
