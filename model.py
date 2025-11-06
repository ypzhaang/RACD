# -*- coding: gbk -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Neural CD + Learnable Q
    """
    def __init__(self, student_n, exer_n, knowledge_n,
                 tau=2.0, thr=0.5, beta=0.5, learnable_beta=False):
        super().__init__()
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim

        # -------- embeddings / params --------
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)        # h^s raw
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)  # k^diff
        self.e_discrimination = nn.Embedding(self.exer_n, 1)               # e^disc

        # -------- learnable Q (logits) --------
        self.q_logits = nn.Parameter(torch.zeros(exer_n, knowledge_n))

        # -------- pred head --------
        self.prednet_full1 = nn.Linear(self.knowledge_dim, 512)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(512, 256)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(256, 1)

        # -------- mixing with given Q --------
        if learnable_beta:
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.register_buffer("beta", torch.tensor(float(beta)))

        # binarization controls
        self.tau = float(tau)
        self.thr = float(thr)

        # init
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() >= 2:
                nn.init.xavier_normal_(p)

    # ----- utilities for Q -----
    def get_q_soft(self):   # [E, K] in (0,1)
        return torch.sigmoid(self.q_logits / self.tau)

    def get_q_bin(self):    # Hard (0/1) with ST
        q_soft = self.get_q_soft()
        q_hard = (q_soft > self.thr).float()
        # straight-through: forward hard, backward soft
        return q_hard + (q_soft - q_soft.detach())

    def get_q_eff(self, exer_id, q_given=None):
        """
        Return the effective Q used in forward for a batch of exercises.
        q_given: [B, K] float in {0,1} or [0,1], may be None
        """
        q_learn_soft = self.get_q_soft()[exer_id]  # [B, K]
        q_learn_bin = (q_learn_soft > self.thr).float()
        q_learn_st = q_learn_bin + (q_learn_soft - q_learn_soft.detach())  # ST

        if q_given is None:
            return q_learn_st
        else:
            beta = self.beta.clamp(0., 1.)
            return beta * q_given + (1. - beta) * q_learn_st

    # ----- forward -----
    def forward(self, stu_id, exer_id, q_given=None, return_stu_emb=False, return_q_soft=False):
        """
        q_given: [B, K] 给定的 Q 行（可选；若 None 则纯用学到的 Q）
        """
        stu_raw = self.student_emb(stu_id)
        stu_emb = torch.sigmoid(stu_raw)                          # [B, K]
        k_diff = torch.sigmoid(self.k_difficulty(exer_id))        # [B, K]
        e_disc = torch.sigmoid(self.e_discrimination(exer_id)) * 10  # [B, 1]

        q_eff = self.get_q_eff(exer_id, q_given=q_given)          # [B, K]

        # IRT-like feature
        x = e_disc * (stu_emb - k_diff) * q_eff                   # [B, K]
        x = self.drop_1(torch.sigmoid(self.prednet_full1(x)))
        x = self.drop_2(torch.sigmoid(self.prednet_full2(x)))
        logits = self.prednet_full3(x)                             # [B, 1]  <-- NO sigmoid here

        outs = [logits]
        if return_stu_emb:
            outs.append(stu_emb)
        if return_q_soft:
            outs.append(self.get_q_soft()[exer_id])
        return tuple(outs) if len(outs) > 1 else logits

    # keep your clipper if needed
    def apply_clipper(self): pass

    # expose for evaluation
    @torch.no_grad()
    def export_Q(self, hard=True):
        q_soft = self.get_q_soft()
        if hard:
            return (q_soft > self.thr).float()
        return q_soft
