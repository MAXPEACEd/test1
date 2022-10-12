import math
import numpy as np
import re
from easydict import EasyDict as edict
import copy
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from models.mytrans import AttFeatTrans, InitWeights
# from attention-is-all-you-need-pytorch-master.transformer.

class PolyformerLayer(InitWeights): #here are net init weights,which will be use.
    def __init__(self, name, config):
        super(PolyformerLayer, self).__init__(config)
        self.name = name
        self.chan_axis = config.chan_axis
        self.feat_dim = config.feat_dim
        self.num_attractors = config.num_attractors
        self.qk_have_bias = config.qk_have_bias
        self.in_trans = AttFeatTrans(config, name + 'in-trans' )
        self.out_trans = AttFeatTrans(config, name + 'out-trans')
        self.attractors = Parameter(torch.randn(1, self.num_attractors, self.feat_dim))
        self.in_feat_layernorm = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=False)
        self.poly_do_layernorm = config.poly_do_layernorm
        print("Polyformer layer {}: {} attractors, {} channels, {} layernorm".format(
            name, self.num_attractors, self.feat_dim,
            'with' if self.poly_do_layernorm else 'no'))

        self.pool2x = nn.AvgPool2d(2) # 2-dim-avg-pooling
        self.apply(self.init_weights) # three function ,should be written in mytrans.
        self.apply(self.tie_qk)
        #self.apply(self.add_identity_bias)

    def forward(self,in_feat):
        '''could downsample, keep it first'''
        B = in_feat.shape[0]
        D = in_feat.shape[1]

        # in_feat is 288x288. Full computation for transformers is a bit slow.
        # So downsample it by 2x. The performance is almost the same.
        # in_feat_half0 = self.pool2x(in_feat)  I think don't need it's downsample the dim
        #in_feat_half = in_feat_half0.transpose(self.chan_axis, -1)
        # Using layernorm reduces performance by 1-2%. Maybe because in_feat has just passed through ReLU(),
        # and a naive layernorm makes zero activations non-zero.
        if self.poly_do_layernorm:
            in_feat_half = self.infeat_norm_layer(in_feat)
        #vfeat = in_feat_half.reshape((B, -1, self.feat_dim))

        batch_attractors = self.attractors.expand(B, -1, -1)
        l_feat =in_feat.view(B, 1, D)
        new_batch_attractors = self.in_trans(batch_attractors, l_feat)  # 50 256 768,50 1 768 =>50 256 768 batch_attractors --c
        vfeat_out = self.out_trans(l_feat, new_batch_attractors) #50 768
        vfeat_out = vfeat_out.transpose(self.chan_axis, -1) #50 768
        out_feat = vfeat_out.reshape(in_feat.shape) #50 768
        #out_feat = F.interpolate(out_feat_half, size=in_feat.shape[2:],
        #                        mode='bilinear', align_corners=False)  #about 采样
        out_feat = in_feat + out_feat

        return out_feat


class Polyformer(nn.Module):
    def __init__(self, feat_dim, chan_axis=1, args=None):
        config = edict()
        if args is None:
            config.num_attractors = 256
            #config.num_modes = 4
            config.tie_qk_scheme = 'loose'
            config.qk_have_bias = True
            #config.pos_code_type = 'lsinu'
        else:
            config.num_attractors = args.num_attractors
            '''
            if args.num_modes != -1:
                config.num_modes = args.num_modes  #num_modes?? segtran里面的不管
            else:
                config.num_modes = 4
            '''
            config.tie_qk_scheme = args.tie_qk_scheme  # shared, loose, or none.
            config.qk_have_bias = args.qk_have_bias
            #config.pos_code_type = args.pos_code_type
        config.num_head = args.num_head
        config.num_layers = 1
        config.in_feat_dim = feat_dim  #input dim. it convey to AttFeaTrans
        config.feat_dim = feat_dim
        config.min_feat_dim = feat_dim  #  i dont't know the difference

        #here are some config
        # Removing biases from V seems to slightly improve performance.
        config.v_has_bias = False
        # has_FFN is False: Only aggregate features, not transform them with an FFN.
        # In the old setting, has_FFN is implicitly True.
        # To reproduce paper results, please set it to True.
        config.has_FFN = False
        #config.attn_clip = 500
        #config.cross_attn_score_scale = 1.
        config.base_initializer_range = 0.02
        #config.hidden_dropout_prob = 0.1
        config.attention_probs_dropout_prob = 0.2
        config.attn_diag_cycles = 500  # print debug info of the attention matrix every 500 iterations.
        #config.ablate_multihead = False
        #config.eval_robustness = False
        config.pool_modes_feat = 'softmax'  # softmax, max, mean, or none.
        config.mid_type = 'shared'  # shared, private, or none.
        config.trans_output_type = 'private'  # shared or private.
        config.act_fun = F.gelu
        #config.feattrans_lin1_idbias_scale = 10
        #config.query_idbias_scale = 10
        config.chan_axis = chan_axis
        #config.use_attn_consist_loss = False
        config.poly_do_layernorm = False

        super(Polyformer, self).__init__()

        polyformer_layers = []
        for i in range(config.num_layers):
            if i > 0:
                config.only_first_linear = False
            polyformer_layers.append(PolyformerLayer(str(i), config) )
        self.polyformer_layers = nn.Sequential(*polyformer_layers)

    def forward(self, in_feat):
            out_feat = self.polyformer_layers(in_feat)
            return out_feat