# -*- coding: gbk -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net

# Read configuration from config.txt
with open('config.txt') as i_f:
    i_f.readline()  # Skip comment line
    student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

# can be changed according to command parameter
device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
epoch_n = 5


alpha_bin = 1e-3

alpha_L1 = 1e-4

alpha_hier = 0.0 

# 结果目录和日志路径
result_dir = None
log_path = None


E_parent_idx = None   # [|E|]
E_child_idx  = None   # [|E|]

C_mat = None  # torch.FloatTensor [K, K]
C_mask_parent_has_child = None  # [K] 0/1
C_child_counts = None  # [K] 子节点数量，避免除0



def load_concept_matrix():
    """
    从 ./data/C.csv 加载概念父子关系矩阵
    要求：矩阵大小 = 知识概念数量 * 知识概念数量；C[i,j]=1 表示 i 是 j 的父概念；对角线为 0
    """
    global C_mat, C_mask_parent_has_child, C_child_counts, E_parent_idx, E_child_idx
    csv_path = os.path.join('.', 'data', 'C.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'未找到 C.csv：{csv_path}')
    np_mat = np.loadtxt(csv_path, delimiter=',')
    if np_mat.shape[0] != knowledge_n or np_mat.shape[1] != knowledge_n:
        raise ValueError(f'C.csv 尺寸不匹配：读取到 {np_mat.shape}，但 knowledge_n={knowledge_n}')
    C_mat = torch.from_numpy(np_mat).float()  # [K, K]

    # 预计算 mask 与子计数
    C_child_counts = C_mat.sum(dim=1)                         # [K]
    C_mask_parent_has_child = (C_child_counts > 0).float()    # [K]
    C_child_counts = torch.clamp(C_child_counts, min=1.0)

    # 预计算父/子边索引，避免 [N,K,K] 扩张
    edges = (C_mat > 0).nonzero(as_tuple=False)               # [|E|, 2], 每行 [p, c]
    if edges.numel() == 0:
        E_parent_idx = torch.empty(0, dtype=torch.long)
        E_child_idx  = torch.empty(0, dtype=torch.long)
    else:
        E_parent_idx = edges[:, 0].long()
        E_child_idx  = edges[:, 1].long()


def build_and_set_dirs():
    global result_dir, log_path, tree_log_path
    result_dir = f'result_alpha_bin{alpha_bin}_alpha_L1{alpha_L1}_alpha_hier{alpha_hier}/' 
    os.makedirs(result_dir, exist_ok=True)
    q_dir  = os.path.join(result_dir, 'Q');  os.makedirs(q_dir, exist_ok=True)
    hs_dir = os.path.join(result_dir, 'hs'); os.makedirs(hs_dir, exist_ok=True)
    log_path = os.path.join(result_dir, 'model_val.txt')
    tree_log_path = os.path.join(result_dir, 'tree_metrics.txt') 

def binarization_loss(hs):
    return (hs * (1.0 - hs)).mean()

def hs_L1_loss(hs):
    return torch.abs(hs).mean()

def concept_hier_loss(stu_emb_batch, delta=0):
    """
    stu_emb_batch: [B, K]，学生掌握度
    delta: margin δ，默认 0.05
    """
    if C_mat is None:
        return torch.tensor(0.0, device=stu_emb_batch.device)

    C = C_mat.to(stu_emb_batch.device)  # [K, K]
    mask = C_mask_parent_has_child.to(stu_emb_batch.device)  # [K]
    child_counts = C_child_counts.to(stu_emb_batch.device)   # [K]

    H = stu_emb_batch  # [B, K]
    child_sum = (H.unsqueeze(1) * C.unsqueeze(0)).sum(dim=2)  # [B, K]
    child_mean = child_sum / child_counts.unsqueeze(0)        # [B, K]

    # 带 δ 的约束：ReLU(child_mean - h_parent - δ)
    viol = torch.relu(child_mean - H - delta)

    if mask.sum() == 0:
        return torch.tensor(0.0, device=stu_emb_batch.device)

    loss = ((viol ** 2) * mask.unsqueeze(0)).sum() / (H.size(0) * mask.sum())
    return loss


# ===================== 树指标（AC, PVR） =====================

@torch.no_grad()
def edge_AC_torch(H, C, delta=0, chunk_size=20000):
  
    # 用预计算的边索引，避免 [N,K,K] 广播
    if E_parent_idx is None or E_parent_idx.numel() == 0:
        return torch.tensor(1.0, device=H.device)

    N = H.size(0)
    total_true = 0
    E = E_parent_idx.numel()

    # 边分块，减少显存峰值
    for start in range(0, E, chunk_size):
        end = min(start + chunk_size, E)
        p_idx = E_parent_idx[start:end].to(H.device)  # [e]
        c_idx = E_child_idx[start:end].to(H.device)   # [e]
        Hp = H[:, p_idx]                               # [N, e]
        Hc = H[:, c_idx]                               # [N, e]
        ok = (Hp - Hc) >= delta                        # [N, e]
        total_true += ok.sum().item()

    ac = total_true / (N * E)
    return torch.tensor(ac, device=H.device)


@torch.no_grad()
def edge_AC_torch(H, C, delta=0, chunk_size=20000):

    if E_parent_idx is None or E_parent_idx.numel() == 0:
        return torch.tensor(1.0, device=H.device)

    N = H.size(0)
    total_true = 0
    E = E_parent_idx.numel()


    for start in range(0, E, chunk_size):
        end = min(start + chunk_size, E)
        p_idx = E_parent_idx[start:end].to(H.device)  # [e]
        c_idx = E_child_idx[start:end].to(H.device)   # [e]
        Hp = H[:, p_idx]                               # [N, e]
        Hc = H[:, c_idx]                               # [N, e]
        ok = (Hp - Hc) >= delta                        # [N, e]
        total_true += ok.sum().item()

    ac = total_true / (N * E)
    return torch.tensor(ac, device=H.device)


@torch.no_grad()
def edge_PVR_cont_torch(H, C, delta=0, chunk_size=20000):

    if E_parent_idx is None or E_parent_idx.numel() == 0:
        return torch.tensor(0.0, device=H.device)

    N = H.size(0)
    total_viol = 0
    E = E_parent_idx.numel()

    for start in range(0, E, chunk_size):
        end = min(start + chunk_size, E)
        p_idx = E_parent_idx[start:end].to(H.device)
        c_idx = E_child_idx[start:end].to(H.device)
        Hp = H[:, p_idx]                                # [N, e]
        Hc = H[:, c_idx]                                # [N, e]
        viol = (Hc - Hp - delta) > 0                    # [N, e]
        total_viol += viol.sum().item()

    pvr = total_viol / (N * E)
    return torch.tensor(pvr, device=H.device)
# ===============================================================


def train():
    data_loader = TrainDataLoader()
    net = Net(student_n, exer_n, knowledge_n)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    print('training model...')

    loss_function = nn.BCELoss()

    hs_dir = os.path.join(result_dir, 'hs')
    os.makedirs(hs_dir, exist_ok=True)

    for epoch in range(epoch_n):
        data_loader.reset()
        running_loss = 0.0
        running_main = 0.0
        running_bin = 0.0
        running_l1 = 0.0
        running_hier = 0.0
        batch_count = 0


        epoch_total_loss_sum = 0.0
        epoch_main_loss_sum = 0.0
        epoch_bin_loss_sum = 0.0
        epoch_l1_loss_sum = 0.0
        epoch_hier_loss_sum = 0.0

        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            input_stu_ids = input_stu_ids.to(device)
            input_exer_ids = input_exer_ids.to(device)
            input_knowledge_embs = input_knowledge_embs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            probs = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs).view(-1)  
            main_loss = loss_function(probs, labels.float())
    
            #hs_batch = net.get_knowledge_status(input_stu_ids)      # [B, K] in [0,1]
            hs_batch = torch.sigmoid(net.student_emb(input_stu_ids))

            bin_loss = binarization_loss(hs_batch)
            l1_loss = hs_L1_loss(hs_batch)

            hier_loss = concept_hier_loss(hs_batch)###########

            loss = main_loss + alpha_bin * bin_loss + alpha_L1 * l1_loss + alpha_hier * hier_loss
            loss.backward()
            optimizer.step()
            net.apply_clipper()

 
            running_loss += loss.item()
            running_main += main_loss.item()
            running_bin += bin_loss.item()
            running_l1 += l1_loss.item()
            running_hier += hier_loss.item()

            epoch_total_loss_sum += loss.item()
            epoch_main_loss_sum += main_loss.item()
            epoch_bin_loss_sum += bin_loss.item()
            epoch_l1_loss_sum += l1_loss.item()
            epoch_hier_loss_sum += hier_loss.item()
            
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f  main: %.3f  bin: %.6f  l1: %.6f  hier: %.6f' %
                      (epoch + 1, batch_count + 1,
                       running_loss / 200, running_main / 200, running_bin / 200, running_l1 / 200, running_hier / 200))
                running_loss = 0.0
                running_main = 0.0
                running_bin = 0.0
                running_l1 = 0.0
                running_hier = 0.0


        with torch.no_grad():
            all_ids = torch.arange(student_n, dtype=torch.long, device=device)
            hs_all = net.get_knowledge_status(all_ids).cpu().numpy()   # [N, K]
            np.save(os.path.join(hs_dir, f'hs����_epoch{epoch + 1}.npy'), hs_all)

        # validate and save current model every epoch
  
        avg_total_loss = epoch_total_loss_sum / max(1, batch_count)
        avg_main_loss  = epoch_main_loss_sum  / max(1, batch_count)
        avg_bin_loss   = epoch_bin_loss_sum   / max(1, batch_count)
        avg_l1_loss    = epoch_l1_loss_sum    / max(1, batch_count)
        avg_hier_loss  = epoch_hier_loss_sum  / max(1, batch_count)

        rmse, auc = validate(net, epoch, avg_total_loss, avg_main_loss, avg_bin_loss, avg_l1_loss, avg_hier_loss)
        save_snapshot(net, os.path.join(result_dir, f'model_epoch{epoch + 1}'))

def validate(model, epoch, avg_total_loss, avg_main_loss, avg_bin_loss, avg_l1_loss, avg_hier_loss):
    data_loader = ValTestDataLoader('validation')
    net = Net(student_n, exer_n, knowledge_n)
    print('validating model...')
    data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids = input_stu_ids.to(device)
        input_exer_ids = input_exer_ids.to(device)
        input_knowledge_embs = input_knowledge_embs.to(device)
        labels = labels.to(device)

        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs).view(-1)  # ֱ�Ӹ���

        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch+1, accuracy, rmse, auc))
    

    with torch.no_grad():
        all_ids = torch.arange(student_n, dtype=torch.long, device=device)
        H = net.get_knowledge_status(all_ids)        # [N, K] (tensor, on device)
        C = C_mat.to(device)
        ac  = edge_AC_torch(H, C, delta=0).item()
        pvr = edge_PVR_cont_torch(H, C, delta=0).item()

    print('epoch= %d, AC= %.6f, PVR= %.6f' % (epoch+1, ac, pvr))

    with open(tree_log_path, 'a', encoding='utf8') as tf:
        tf.write('epoch= %d, AC= %.6f, PVR= %.6f\n' % (epoch+1, ac, pvr))
    # -----------------------------------------------------------------------
    

    with open(log_path, 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f, total_loss= %f, main_loss= %f, bin_loss= %f, hs_L1_loss= %f, hier_loss= %f\n' % 
                (epoch+1, accuracy, rmse, auc, avg_total_loss, avg_main_loss, avg_bin_loss, avg_l1_loss, avg_hier_loss))

    return rmse, auc

def save_snapshot(model, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()

if __name__ == '__main__':
        if (len(sys.argv) != 6) or ((sys.argv[1] != 'cpu') and ('cuda:' not in sys.argv[1])) or (not sys.argv[2].isdigit()) or (not sys.argv[3].replace('.', '').isdigit()) or (not sys.argv[4].replace('.', '').isdigit()) or (not sys.argv[5].replace('.', '').isdigit()):
        print('command:\n\tpython train_hs.py {device} {epoch} {alpha_bin} {alpha_L1} {alpha_hier}\nexample:\n\tpython train_hs.py cuda:0 70 0.001 0.0001 0.01')
        exit(1)
    else:
        device = torch.device(sys.argv[1])
        epoch_n = int(sys.argv[2])
        alpha_bin = float(sys.argv[3])
        alpha_L1 = float(sys.argv[4])
        alpha_hier = float(sys.argv[5])


    load_concept_matrix()

    build_and_set_dirs()

    train()