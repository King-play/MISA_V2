import numpy as np
import random
import math

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig

from utils import to_gpu
from utils import ReverseLayerF


def masked_mean(tensor, mask, dim):
    #Finding the mean along dim
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)

def masked_max(tensor, mask, dim):
    #Finding the max along dim
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf
    return (masked + neg_inf).max(dim=dim)


# 时空解耦模块
class SpatialTemporalDecouple(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        时空解耦模块
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        """
        super(SpatialTemporalDecouple, self).__init__()
        
        # 时间序列处理部分
        self.temporal_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.temporal_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 空间序列处理部分 (使用自注意力机制)
        self.spatial_attention = nn.MultiheadAttention(input_dim, num_heads=4, batch_first=True)
        self.spatial_fc = nn.Linear(input_dim, hidden_dim)
        
        # 归一化层
        self.layer_norm_temporal = nn.LayerNorm(hidden_dim)
        self.layer_norm_spatial = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        x: 输入特征 [batch_size, seq_len, feature_dim]
        返回: 时间序列特征和空间序列特征
        """
        batch_size, seq_len, _ = x.size()
        
        # 提取时间序列特征
        temporal_output, _ = self.temporal_lstm(x)
        temporal_output = self.temporal_fc(temporal_output)
        temporal_output = self.layer_norm_temporal(temporal_output)
        
        # 提取空间序列特征
        spatial_output, _ = self.spatial_attention(x, x, x)
        spatial_output = self.spatial_fc(spatial_output)
        spatial_output = self.layer_norm_spatial(spatial_output)
        
        return temporal_output, spatial_output


# 增强的交叉注意力融合模块
class EnhancedCrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, expand_factor=4):
        """
        增强的交叉注意力融合模块
        hidden_dim: 隐藏层维度
        num_heads: 注意力头数
        expand_factor: 序列扩展因子，增加虚拟序列长度
        """
        super(EnhancedCrossAttentionFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.expand_factor = expand_factor
        
        # 特征扩展层 - 将单个向量扩展为多个不同的表示
        self.temporal_expander = nn.Linear(hidden_dim, hidden_dim * expand_factor)
        self.spatial_expander = nn.Linear(hidden_dim, hidden_dim * expand_factor)
        
        # 位置编码
        self.positional_encoding = nn.Parameter(torch.randn(expand_factor, hidden_dim))
        
        # 多头交叉注意力
        self.cross_attention_ts = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        self.cross_attention_st = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        self.cross_attention_joint = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        
        # 自注意力层增强表示
        self.self_attention_temporal = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        self.self_attention_spatial = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        
        # 联合表示投影
        self.joint_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 门控融合机制
        self.gate_temporal = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_spatial = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_joint = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 输出投影 - 增加更多的交叉注意力组合
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)
        
        # 残差连接的权重
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.5)
        
    def expand_features(self, features, expander):
        """
        将单个特征向量扩展为多个不同的表示
        """
        batch_size = features.size(0)
        
        # 扩展特征
        expanded = expander(features.squeeze(1))  # [batch_size, hidden_dim * expand_factor]
        expanded = expanded.view(batch_size, self.expand_factor, self.hidden_dim)
        
        # 添加位置编码
        expanded = expanded + self.positional_encoding.unsqueeze(0)
        
        return expanded
    
    def forward(self, temporal_features, spatial_features):
        """
        temporal_features: 时间特征 [batch_size, 1, hidden_dim]
        spatial_features: 空间特征 [batch_size, 1, hidden_dim]
        返回: 融合后的特征 [batch_size, 1, hidden_dim]
        """
        batch_size = temporal_features.size(0)
        
        # 扩展特征到更长的序列
        temporal_expanded = self.expand_features(temporal_features, self.temporal_expander)  # [batch_size, expand_factor, hidden_dim]
        spatial_expanded = self.expand_features(spatial_features, self.spatial_expander)
        
        # 自注意力增强各自的表示
        temporal_enhanced, _ = self.self_attention_temporal(temporal_expanded, temporal_expanded, temporal_expanded)
        spatial_enhanced, _ = self.self_attention_spatial(spatial_expanded, spatial_expanded, spatial_expanded)
        
        temporal_enhanced = self.layer_norm1(temporal_enhanced + temporal_expanded)
        spatial_enhanced = self.layer_norm2(spatial_enhanced + spatial_expanded)
        
        # 创建联合表示
        joint_features = torch.cat([
            temporal_enhanced.mean(dim=1, keepdim=True), 
            spatial_enhanced.mean(dim=1, keepdim=True)
        ], dim=-1)
        joint_features = self.joint_projection(joint_features)
        joint_expanded = joint_features.expand(-1, self.expand_factor, -1)
        
        # 多种交叉注意力计算
        # 1. 时间特征关注空间特征
        cross_ts, _ = self.cross_attention_ts(temporal_enhanced, spatial_enhanced, spatial_enhanced)
        
        # 2. 空间特征关注时间特征
        cross_st, _ = self.cross_attention_st(spatial_enhanced, temporal_enhanced, temporal_enhanced)
        
        # 3. 时间特征关注联合表示
        cross_tj, _ = self.cross_attention_joint(temporal_enhanced, joint_expanded, joint_expanded)
        
        # 4. 空间特征关注联合表示
        cross_sj, _ = self.cross_attention_joint(spatial_enhanced, joint_expanded, joint_expanded)
        
        # 5. 联合表示关注时间特征
        cross_jt, _ = self.cross_attention_joint(joint_expanded, temporal_enhanced, temporal_enhanced)
        
        # 6. 联合表示关注空间特征
        cross_js, _ = self.cross_attention_joint(joint_expanded, spatial_enhanced, spatial_enhanced)
        
        # 门控融合机制
        gate_input_t = torch.cat([cross_ts.mean(dim=1), cross_tj.mean(dim=1)], dim=-1)
        gate_input_s = torch.cat([cross_st.mean(dim=1), cross_sj.mean(dim=1)], dim=-1)
        gate_input_j = torch.cat([cross_jt.mean(dim=1), cross_js.mean(dim=1)], dim=-1)
        
        gate_t = torch.sigmoid(self.gate_temporal(gate_input_t))
        gate_s = torch.sigmoid(self.gate_spatial(gate_input_s))
        gate_j = torch.sigmoid(self.gate_joint(gate_input_j))
        
        # 应用门控
        weighted_ts = gate_t * cross_ts.mean(dim=1)
        weighted_st = gate_s * cross_st.mean(dim=1)
        weighted_tj = gate_t * cross_tj.mean(dim=1)
        weighted_sj = gate_s * cross_sj.mean(dim=1)
        weighted_jt = gate_j * cross_jt.mean(dim=1)
        weighted_js = gate_j * cross_js.mean(dim=1)
        
        # 拼接所有加权的交叉注意力输出
        fused_features = torch.cat([
            weighted_ts, weighted_st, weighted_tj, 
            weighted_sj, weighted_jt, weighted_js
        ], dim=-1)
        
        # 输出投影
        fused_features = self.output_projection(fused_features)
        fused_features = self.layer_norm3(fused_features)
        
        # 残差连接
        original_input = (temporal_features.squeeze(1) + spatial_features.squeeze(1)) / 2
        fused_features = self.residual_weight * fused_features + (1 - self.residual_weight) * original_input
        
        return fused_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]


# let's define a simple model that can deal with multimodal variable length sequence
class MISA(nn.Module):
    def __init__(self, config):
        super(MISA, self).__init__()

        self.config = config
        self.text_size = config.embedding_size
        self.visual_size = config.visual_size
        self.acoustic_size = config.acoustic_size

        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = self.config.activation()
        self.tanh = nn.Tanh()
        
        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU
        # defining modules - two layer bidirectional LSTM with layer norm in between

        if self.config.use_bert:
            # Initializing a BERT bert-base-uncased style configuration
            bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        else:
            self.embed = nn.Embedding(len(config.word2id), input_sizes[0])
            self.trnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
            self.trnn2 = rnn(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)
        
        self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        
        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)

        ##########################################
        # mapping modalities to same sized space
        ##########################################
        if self.config.use_bert:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))
        else:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=hidden_sizes[0]*4, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=hidden_sizes[1]*4, out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_sizes[2]*4, out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))

        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
        
        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        ##########################################
        # 时空解耦和增强交叉注意力融合模块
        ##########################################
        # 对共享空间进行时空解耦
        self.spatial_temporal_decouple = SpatialTemporalDecouple(
            input_dim=config.hidden_size, 
            hidden_dim=config.hidden_size
        )
        
        # 改用增强的交叉注意力融合
        self.enhanced_cross_attention_fusion = EnhancedCrossAttentionFusion(
            hidden_dim=config.hidden_size,
            num_heads=8,  # 增加注意力头数
            expand_factor=4  # 扩展因子
        )

        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))

        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not self.config.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
            self.discriminator.add_module('discriminator_layer_2', nn.Linear(in_features=config.hidden_size, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################
        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=4))

        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size*6, out_features=self.config.hidden_size*3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.config.hidden_size*3, out_features= output_size))

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths)

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)

        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2

    def alignment(self, sentences, visual, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        batch_size = lengths.size(0)
        
        if self.config.use_bert:
            bert_output = self.bertmodel(input_ids=bert_sent, 
                                         attention_mask=bert_sent_mask, 
                                         token_type_ids=bert_sent_type)      
            bert_output = bert_output[0]
            # masked mean
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)  
            bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
            utterance_text = bert_output
        else:
            # extract features from text modality
            sentences = self.embed(sentences)
            final_h1t, final_h2t = self.extract_features(sentences, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
            utterance_text = torch.cat((final_h1t, final_h2t), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # Shared-private encoders
        self.shared_private(utterance_text, utterance_video, utterance_audio)

        if not self.config.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.config.reverse_grad_weight)

            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None

        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator( (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a)/3.0 )
        
        # For reconstruction
        self.reconstruct()
        
        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v,  self.utt_shared_a), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        o = self.fusion(h)
        return o
    
    def reconstruct(self,):
        # 重构使用融合后的共享表示
        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)

    def shared_private(self, utterance_t, utterance_v, utterance_a):
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)

        # Private-shared components - 私有空间保持不变
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        # 获取共享空间的原始表示
        shared_t = self.shared(utterance_t)
        shared_v = self.shared(utterance_v)
        shared_a = self.shared(utterance_a)
        
        # 只对共享空间进行时空解耦
        # 将向量扩展为序列形式 [batch_size, 1, hidden_dim]
        batch_size = utterance_t.size(0)
        shared_t_seq = shared_t.view(batch_size, 1, -1)
        shared_v_seq = shared_v.view(batch_size, 1, -1)
        shared_a_seq = shared_a.view(batch_size, 1, -1)
        
        # 对每个模态的共享表示进行时空解耦
        t_temporal, t_spatial = self.spatial_temporal_decouple(shared_t_seq)
        v_temporal, v_spatial = self.spatial_temporal_decouple(shared_v_seq)
        a_temporal, a_spatial = self.spatial_temporal_decouple(shared_a_seq)
        
        # 保存时间和空间特征用于计算损失
        self.t_temporal = t_temporal
        self.t_spatial = t_spatial
        self.v_temporal = v_temporal
        self.v_spatial = v_spatial
        self.a_temporal = a_temporal
        self.a_spatial = a_spatial
        
        # 对每个模态的时间和空间特征进行增强交叉注意力融合
        t_fused = self.enhanced_cross_attention_fusion(t_temporal, t_spatial).squeeze(1)  # [batch_size, hidden_dim]
        v_fused = self.enhanced_cross_attention_fusion(v_temporal, v_spatial).squeeze(1)
        a_fused = self.enhanced_cross_attention_fusion(a_temporal, a_spatial).squeeze(1)
        
        # 将融合后的特征作为最终的共享表示
        self.utt_shared_t = t_fused
        self.utt_shared_v = v_fused
        self.utt_shared_a = a_fused

    def forward(self, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        batch_size = lengths.size(0)
        o = self.alignment(sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask)
        return o
