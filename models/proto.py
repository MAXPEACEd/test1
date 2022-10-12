import sys
import os
current_dir = os.path.abspath(os.path.dirname('proto.py'))
sys.path.append(current_dir)
sys.path.append("..")
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from polyformer import Polyformer

class Proto(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, dot=False, polyformer_args=None):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.dot = dot

        self.use_polyformer = (polyformer_args is not None) and \
                              (polyformer_args.polyformer_mode is not None)

        if self.use_polyformer:
            self.polyformer = Polyformer(feat_dim=768, args=polyformer_args) #the pra should change

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, N, K, total_Q):
        """
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        """
        #test = torch.cuda.current_device()
        support_emb = self.sentence_encoder(support)  # 应该是经过encoder之后，因为本来support是没有D的。(B * N * K, D), where D is the hidden size
        query_emb = self.sentence_encoder(query)  # (B * total_Q, D)

        if self.use_polyformer:
            support_emb = self.polyformer(support_emb)
            query_emb = self.polyformer(query_emb)

        hidden_size = support_emb.size(-1)
        support = self.drop(support_emb)
        query = self.drop(query_emb)
        support = support.view(-1, N, K, hidden_size)  # (B, N, K, D)
        query = query.view(-1, total_Q, hidden_size)  # (B, total_Q, D)

        # Prototypical Networks 
        # Ignore NA policy
        support = torch.mean(support, 2)  # Calculate prototype for each class
        logits = self.__batch_dist__(support, query)  # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)  # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        return logits, pred
