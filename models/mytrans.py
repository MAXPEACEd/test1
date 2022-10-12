import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#from  attention_is_all_you_need_pytorch_master.transformer.Modules import ScaledDotProductAttention
#from attention_is_all_you_need_pytorch_master.transformer.SubLayers import MultiHeadAttention
torch.set_printoptions(sci_mode=False)

class AttFeatTrans(nn.Module):
    def __init__(self, config, name):
        super(AttFeatTrans,self).__init__()
        self.config = config
        self.name = name
        self.head = config.num_head
        # some setting
        self.in_feat_dim = config.in_feat_dim
        self.feat_dim = config.feat_dim  # don't know where it is manifested since then
        self.query = nn.Linear(self.in_feat_dim, self.in_feat_dim, bias=config.qk_have_bias)
        self.key = nn.Linear(self.in_feat_dim, self.in_feat_dim, bias=config.qk_have_bias)
        self.value = nn.Linear(self.in_feat_dim, self.in_feat_dim, bias = config.v_has_bias)
        #self.out_trans = MultiHeadAttention(self.head, self.in_feat_dim * self.head, self.in_feat_dim, self.in_feat_dim, dropout=0.1) #some dim in it
        self.att_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.feat_dim_onehead = self.in_feat_dim // self.head

        self.intermediate = config.act_fun #gelu
        self.tie_qk_scheme = config.tie_qk_scheme  #not use it first
        print("{} in_feat_dim: {}, feat_dim: {}, qk_have_bias: {}".format(
        self.name, self.in_feat_dim, self.feat_dim, config.qk_have_bias))

    def tie_qk(self, tie_qk_scheme=None):
        # override config.tie_qk_scheme
        if tie_qk_scheme is not None:
            self.tie_qk_scheme = tie_qk_scheme

        print("Initialize QK scheme: {}".format(self.tie_qk_scheme))
        if self.tie_qk_scheme == 'shared':
            self.key.weight = self.query.weight
            if self.key.bias is not None:
                self.key.bias = self.query.bias

        elif self.tie_qk_scheme == 'loose':
            self.key.weight.data.copy_(self.query.weight)
            if self.key.bias is not None:
                self.key.bias.data.copy_(self.query.bias)

    def transpose_for_scores(self, x):
        x_new_shape = x.size()[:-1] + (self.head, -1)
        x = x.view(*x_new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, in_query, in_key=None, pos_biases=None): #[50,256,768] [50,1,768]
        if in_key is None:
            in_key = in_query


        mixed_query_feat = self.query(in_query) #[50,256,768]
        mixed_key_feat = self.key(in_key)       #[50,1,768]
        v_first_feat = self.value(in_key)       #[50,1,768]#q,v will same?  infact the dim of qv not same
        '''
        n_head = self.head
        B = in_query.shape[0]
        D = in_query.shape[2]
        q = mixed_query_feat.view(B, n_head , -1 , D)
        k = mixed_key_feat.view(B, n_head, -1, D)
        v = mixed_value_feat.view(B, n_head, -1, D)
        '''
        q = self.transpose_for_scores(mixed_query_feat) #[50 2 256 384]
        k = self.transpose_for_scores(mixed_key_feat)   #[50 2 1 384]
        att_score = torch.matmul(q,k.transpose(-1, -2))
        #output = self.out_trans(mixed_query_feat,mixed_key_feat,mixed_value_feat)
        att_score = att_score / math.sqrt(q.shape[3])  #[50 2 256 1]
        attention_probs = F.softmax(att_score, dim=-1)
        attention_probs = self.att_dropout(attention_probs) #[50 2 256 1]

        # about v(f)
        v_first_feat = v_first_feat.permute(0, 2, 1)  #[50 786 1]
        shape_4d = (v_first_feat.shape[0], self.head, self.feat_dim_onehead, v_first_feat.shape[2]) # [50 2 384 1]
        v_first_feat_4d = v_first_feat.view(shape_4d).permute([0, 1, 3, 2]) #[50 2 1 384]
        v_first_feat_fusion = torch.matmul(attention_probs, v_first_feat_4d) #[50, 2, 256, 384]
        #v_first_feat_fusion_3d = v_first_feat_fusion.permute([0, 1, 3, 2]).reshape(v_first_feat.shape)
        v_first_feat_fusion_3d = v_first_feat_fusion.permute([0, 1, 3, 2]).reshape(v_first_feat.shape[0],v_first_feat.shape[1], -1) # 50 768 256
        v_first_feat = v_first_feat_fusion_3d.permute([0,2,1]) #50 256 768

        v_mid_feat = self.intermediate(v_first_feat) #gelu #50 256 768
        v_last_feat = v_mid_feat + v_first_feat #res #50 256 768
        v_trans_feat = v_last_feat  #50 256 768
        trans_feat = v_trans_feat.squeeze(1) #50 256 768
        return trans_feat

class InitWeights(nn.Module):
    def __init__(self, config, *inputs, **kwargs):
        super(InitWeights, self).__init__()
        self.config = config
        self.act_fun = F.gelu

    def init_weights(self, module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if (np.array(module.weight.shape) < self.config.min_feat_dim).all():
                    print("Skip init of Linear weight %s" % (list(module.weight.shape)))
                else:
                    base_initializer_range = self.config.base_initializer_range
                    module.weight.data.normal_(mean=0.0, std=base_initializer_range)
                # Slightly different from the TF version which uses truncated_normal
                # for initialization cf https://github.com/pytorch/pytorch/pull/5617
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def tie_qk(self, module):
            if isinstance(module, AttFeatTrans) and module.tie_qk_scheme != 'none':
                module.tie_qk()
    '''
        def add_identity_bias(self, module):
            if isinstance(module, AttFeatTrans) or isinstance(module, ExpandedFeatTrans):
                module.add_identity_bias()
     '''
