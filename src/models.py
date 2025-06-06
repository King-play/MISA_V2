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

class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, dropout=0.2, layers=2):
        """
        时序卷积网络实现
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        kernel_size: 卷积核大小
        dropout: Dropout率
        layers: TCN层数
        """
        super(TemporalConvNet, self).__init__()
        
        self.tcn_layers = nn.ModuleList()
        
        # 第一层从输入维度到隐藏维度
        self.tcn_layers.append(nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ))
        
        # 添加后续层
        for i in range(1, layers):
            self.tcn_layers.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=(kernel_size-1)//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
    
    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        return: [batch_size, seq_len, hidden_dim]
        """
        # 将输入转换为TCN所需的格式 [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        # 通过TCN层
        for layer in self.tcn_layers:
            x = layer(x)
        
        # 转回原始格式 [batch_size, seq_len, hidden_dim]
        x = x.transpose(1, 2)
        
        return x

class SpatialTemporalDecouple(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        时空解耦模块
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        """
        
        super(SpatialTemporalDecouple, self).__init__()
        
        # 时间序列处理部分 - 确保模型知道正确的输入维度
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.temporal_lstm = nn.LSTM(input_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.temporal_fc = nn.Linear(hidden_dim, hidden_dim)
        
        # 空间序列处理部分 (使用自注意力机制)
        # 对于自注意力，确保多头数量与隐藏维度兼容
        num_heads = 4
        if input_dim % num_heads != 0:
            # 调整头数以适应输入维度
            num_heads = 1
        
        self.spatial_attention = nn.MultiheadAttention(input_dim, num_heads=num_heads, batch_first=True)
        self.spatial_fc = nn.Linear(input_dim, hidden_dim)
        
        # 归一化层
        self.layer_norm_temporal = nn.LayerNorm(hidden_dim)
        self.layer_norm_spatial = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        x: 输入特征 [batch_size, seq_len, feature_dim]
        返回: 时间序列特征和空间序列特征
        """
        
        batch_size, seq_len, feature_dim = x.size()
        
        # 检查输入维度是否与预期一致
        if feature_dim != self.input_dim:
            # 如果不一致，调整输入
            #print(f"Warning: Input dimension {feature_dim} doesn't match expected dimension {self.input_dim}.")
            # 使用线性投影调整维度
            if feature_dim > self.input_dim:
                x = x[:, :, :self.input_dim]
            else:
                padding = torch.zeros(batch_size, seq_len, self.input_dim - feature_dim, device=x.device)
                x = torch.cat([x, padding], dim=-1)
        
        # 提取时间序列特征
        temporal_output, _ = self.temporal_lstm(x)
        temporal_output = self.temporal_fc(temporal_output)
        temporal_output = self.layer_norm_temporal(temporal_output)
        
        # 提取空间序列特征 (将序列维度视为空间维度)
        spatial_output, _ = self.spatial_attention(x, x, x)
        spatial_output = self.spatial_fc(spatial_output)
        spatial_output = self.layer_norm_spatial(spatial_output)
        
        return temporal_output, spatial_output

class AlignmentFusion(nn.Module):
    def __init__(self, hidden_dim):
        """
        对齐和融合模块
        hidden_dim: 隐藏层维度
        """
        
        super(AlignmentFusion, self).__init__()
        
        # 对齐部分 - 确保多头数量与隐藏维度兼容
        num_heads = 4
        if hidden_dim % num_heads != 0:
            # 调整头数以适应隐藏维度
            num_heads = 1
            
        self.alignment = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        
        # 融合部分
        self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, temporal_features, spatial_features):
        """
        temporal_features: 时间序列特征 [batch_size, seq_len, hidden_dim]
        spatial_features: 空间序列特征 [batch_size, seq_len, hidden_dim]
        返回: 融合后的特征
        """
        
        batch_size, seq_len, _ = temporal_features.size()
        
        # 对齐
        aligned_temporal, _ = self.alignment(temporal_features, spatial_features, spatial_features)
        
        # 融合
        fused_features = torch.cat([aligned_temporal, spatial_features], dim=-1)
        fused_features = self.fusion_fc(fused_features)
        fused_features = self.layer_norm(fused_features)
        
        return fused_features

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
        
        # 文本特征提取器
        if hasattr(config, 'text_extractor'):
            self.text_extractor = config.text_extractor
        else:
            self.text_extractor = 'bert' if self.config.use_bert else 'lstm'
        
        if self.text_extractor == 'bert':
            # Initializing a BERT bert-base-uncased style configuration
            bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        else:  # lstm
            self.embed = nn.Embedding(len(config.word2id), input_sizes[0])
            self.trnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
            self.trnn2 = rnn(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)
        
        # 视觉特征提取器
        if hasattr(config, 'visual_extractor'):
            self.visual_extractor = config.visual_extractor
        else:
            self.visual_extractor = 'lstm'
        
        if self.visual_extractor == 'lstm':
            self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
            self.vrnn2 = rnn(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        else:  # tcn
            # TCN输出维度直接设置为hidden_sizes[1]，这样平均池化和最大池化后为hidden_sizes[1]*2
            self.vtcn = TemporalConvNet(
                input_sizes[1], 
                hidden_sizes[1],  # 修改这里：直接设置为hidden_sizes[1]
                kernel_size=config.tcn_kernel_size if hasattr(config, 'tcn_kernel_size') else 3,
                dropout=config.tcn_dropout if hasattr(config, 'tcn_dropout') else 0.2,
                layers=config.tcn_layers if hasattr(config, 'tcn_layers') else 2
            )
            # 全局特征提取层：输入维度为hidden_sizes[1]*2（平均+最大池化），输出为hidden_sizes[1]*4
            self.v_global_feature = nn.Linear(hidden_sizes[1] * 2, hidden_sizes[1] * 4)
        
        # 声学特征提取器
        if hasattr(config, 'acoustic_extractor'):
            self.acoustic_extractor = config.acoustic_extractor
        else:
            self.acoustic_extractor = 'lstm'
        
        if self.acoustic_extractor == 'lstm':
            self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
            self.arnn2 = rnn(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)
        else:  # tcn
            # TCN输出维度直接设置为hidden_sizes[2]，这样平均池化和最大池化后为hidden_sizes[2]*2
            self.atcn = TemporalConvNet(
                input_sizes[2], 
                hidden_sizes[2],  # 修改这里：直接设置为hidden_sizes[2]
                kernel_size=config.tcn_kernel_size if hasattr(config, 'tcn_kernel_size') else 3,
                dropout=config.tcn_dropout if hasattr(config, 'tcn_dropout') else 0.2,
                layers=config.tcn_layers if hasattr(config, 'tcn_layers') else 2
            )
            # 全局特征提取层：输入维度为hidden_sizes[2]*2（平均+最大池化），输出为hidden_sizes[2]*4
            self.a_global_feature = nn.Linear(hidden_sizes[2] * 2, hidden_sizes[2] * 4)

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
        # 时空解耦和对齐融合模块
        ##########################################
        # 对共享空间进行时空解耦 - 确保使用正确的维度
        self.spatial_temporal_decouple = SpatialTemporalDecouple(
            input_dim=config.hidden_size, 
            hidden_dim=config.hidden_size
        )
        
        # 对时空特征进行对齐和融合
        self.alignment_fusion = AlignmentFusion(
            hidden_dim=config.hidden_size
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

        # 更新Transformer编码器，确保使用batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size, 
            nhead=2,
            batch_first=True  # 添加这个参数
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def extract_lstm_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        """
        使用LSTM提取特征
        
        Args:
            sequence: 输入序列
            lengths: 序列长度
            rnn1, rnn2: LSTM模型
            layer_norm: 层归一化
            
        Returns:
            utterance: 维度为 [batch_size, hidden_size*4] 的特征向量
        """
        batch_size = sequence.size(0)
        
        # 确保lengths与sequence的批次大小一致
        seq_len = sequence.size(1)
        
        # 安全地获取有效长度
        if lengths.size(0) >= batch_size:
            valid_lengths = lengths[:batch_size]
        else:
            # 如果lengths不够长，用最大序列长度填充
            valid_lengths = torch.full((batch_size,), seq_len, dtype=lengths.dtype, device=lengths.device)
            valid_lengths[:lengths.size(0)] = lengths
        
        # 防止lengths中有0
        valid_lengths = torch.clamp(valid_lengths, min=1, max=seq_len)
        
        packed_sequence = pack_padded_sequence(sequence, valid_lengths, batch_first=True, enforce_sorted=False)

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1, batch_first=True)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, valid_lengths, batch_first=True, enforce_sorted=False)

        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)
            
        # 转换为向量 [batch_size, hidden_size*4]
        # final_h1 和 final_h2 的形状是 [num_layers*num_directions, batch_size, hidden_size]
        # 对于双向LSTM，num_directions=2
        final_h1 = final_h1.permute(1, 0, 2).contiguous().view(batch_size, -1)  # [batch_size, num_layers*num_directions*hidden_size]
        final_h2 = final_h2.permute(1, 0, 2).contiguous().view(batch_size, -1)  # [batch_size, num_layers*num_directions*hidden_size]
        
        utterance = torch.cat((final_h1, final_h2), dim=1)  # [batch_size, hidden_size*4]
        
        return utterance

    def extract_tcn_features(self, sequence, lengths, tcn_model, global_feature_layer):
        """
        使用TCN提取特征并转换为固定维度向量
        
        Args:
            sequence: 输入序列 [batch_size, seq_len, input_dim]
            lengths: 序列长度
            tcn_model: TCN模型
            global_feature_layer: 用于生成全局特征的线性层
            
        Returns:
            utterance: 维度为 [batch_size, hidden_size*4] 的特征向量
        """
        original_batch_size = sequence.size(0)
        seq_len = sequence.size(1)
        
        # 安全地获取有效长度
        if lengths.size(0) >= original_batch_size:
            valid_lengths = lengths[:original_batch_size]
        else:
            # 如果lengths不够长，用最大序列长度填充
            valid_lengths = torch.full((original_batch_size,), seq_len, dtype=lengths.dtype, device=lengths.device)
            valid_lengths[:lengths.size(0)] = lengths
        
        # 防止lengths中有0
        valid_lengths = torch.clamp(valid_lengths, min=1, max=seq_len)
        
        # 应用TCN - 获取序列级特征
        tcn_output = tcn_model(sequence)  # [batch_size, seq_len, hidden_dim]
        
        # 使用全局平均池化和最大池化
        pooled_features = []
        for i in range(original_batch_size):  # 使用原始批次大小
            # 获取有效序列长度，确保不越界
            valid_len = min(valid_lengths[i].item(), seq_len)
            
            # 如果有效长度为0，使用零向量
            if valid_len == 0:
                avg_pooled = torch.zeros(tcn_output.size(-1), device=sequence.device)
                max_pooled = torch.zeros(tcn_output.size(-1), device=sequence.device)
            else:
                # 对有效部分求平均和最大值
                valid_output = tcn_output[i, :valid_len]  # [valid_len, hidden_dim]
                avg_pooled = valid_output.mean(dim=0)  # [hidden_dim]
                max_pooled = valid_output.max(dim=0)[0]  # [hidden_dim]
            
            # 组合平均池化和最大池化结果
            combined = torch.cat([avg_pooled, max_pooled], dim=0)  # [hidden_dim * 2]
            pooled_features.append(combined)
        
        # 堆叠所有样本的特征 - 确保批次大小与输入一致
        pooled_features = torch.stack(pooled_features, dim=0)  # [original_batch_size, hidden_dim * 2]
        
        # 应用全局特征层将特征转换为所需维度
        utterance = global_feature_layer(pooled_features)  # [original_batch_size, hidden_size*4]
        
        return utterance

    def alignment(self, sentences, visual, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        # 获取实际的输入批次大小
        #print(f"Debug input shapes: sentences={sentences.shape if sentences is not None else None}")
        #print(f"Debug input shapes: visual={visual.shape}")
        #print(f"Debug input shapes: acoustic={acoustic.shape}")
        #print(f"Debug input shapes: bert_sent={bert_sent.shape}")
        #print(f"Debug input shapes: bert_sent_mask={bert_sent_mask.shape}")
        #print(f"Debug input shapes: bert_sent_type={bert_sent_type.shape}")
        
        # 找出BERT相关张量的实际批次大小
        bert_actual_batch = bert_sent.size(0)
        visual_batch_size = visual.size(0)
        acoustic_batch_size = acoustic.size(0)
        
        #print(f"Debug batch sizes: bert={bert_actual_batch}, visual={visual_batch_size}, acoustic={acoustic_batch_size}")
        
        # 使用BERT数据的实际批次大小，因为文本是主要的模态
        batch_size = bert_actual_batch
        
        #print(f"Debug: Using batch_size={batch_size} based on BERT input")
        
        # 提取文本特征
        if self.text_extractor == 'bert':
            # 使用完整的BERT输入（不截取）
            #print(f"Debug BERT: Processing full BERT input with batch_size={bert_actual_batch}")
            
            bert_output = self.bertmodel(input_ids=bert_sent, 
                                         attention_mask=bert_sent_mask, 
                                         token_type_ids=bert_sent_type)      

            bert_output = bert_output[0]
            #print(f"Debug BERT: bert_output shape={bert_output.shape}")

            # masked mean
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)  
            # 防止除以0
            mask_len = torch.clamp(mask_len, min=1)
            bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
            #print(f"Debug BERT: final bert_output shape={bert_output.shape}")

            utterance_text = bert_output
        else:
            # extract features from text modality using LSTM
            sentences_batch = sentences[:batch_size]
            sentences_batch = self.embed(sentences_batch)
            utterance_text = self.extract_lstm_features(sentences_batch, lengths, self.trnn1, self.trnn2, self.tlayer_norm)

        #print(f"Debug alignment: utterance_text shape={utterance_text.shape}")

        # 为了保持一致性，对视觉和声学数据也使用相同的批次大小
        # 如果视觉/声学数据的批次大小大于BERT的批次大小，我们截取前N个样本
        # 如果小于，我们重复最后的样本来填充（这是一种妥协方案）
        
        # 处理视觉数据
        if visual_batch_size >= batch_size:
            visual_batch = visual[:batch_size]
        else:
            # 如果视觉数据不够，重复最后的样本
            #print(f"Warning: Visual batch size {visual_batch_size} < required {batch_size}, padding with last sample")
            visual_batch = visual
            # 重复最后一个样本来填充
            last_sample = visual[-1:].repeat(batch_size - visual_batch_size, 1, 1)
            visual_batch = torch.cat([visual_batch, last_sample], dim=0)
        
        #print(f"Debug alignment: visual_batch shape={visual_batch.shape}")
        
        if self.visual_extractor == 'lstm':
            utterance_video = self.extract_lstm_features(visual_batch, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        else:  # tcn
            utterance_video = self.extract_tcn_features(visual_batch, lengths, self.vtcn, self.v_global_feature)

        #print(f"Debug alignment: utterance_video shape={utterance_video.shape}")

        # 处理声学数据
        if acoustic_batch_size >= batch_size:
            acoustic_batch = acoustic[:batch_size]
        else:
            # 如果声学数据不够，重复最后的样本
            #print(f"Warning: Acoustic batch size {acoustic_batch_size} < required {batch_size}, padding with last sample")
            acoustic_batch = acoustic
            # 重复最后一个样本来填充
            last_sample = acoustic[-1:].repeat(batch_size - acoustic_batch_size, 1, 1)
            acoustic_batch = torch.cat([acoustic_batch, last_sample], dim=0)
        
        #print(f"Debug alignment: acoustic_batch shape={acoustic_batch.shape}")
        
        if self.acoustic_extractor == 'lstm':
            utterance_audio = self.extract_lstm_features(acoustic_batch, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        else:  # tcn
            utterance_audio = self.extract_tcn_features(acoustic_batch, lengths, self.atcn, self.a_global_feature)

        #print(f"Debug alignment: utterance_audio shape={utterance_audio.shape}")

        # 在调用shared_private之前确保所有特征的批次大小一致
        assert utterance_text.size(0) == batch_size, f"Text utterance batch size {utterance_text.size(0)} != {batch_size}"
        assert utterance_video.size(0) == batch_size, f"Video utterance batch size {utterance_video.size(0)} != {batch_size}"
        assert utterance_audio.size(0) == batch_size, f"Audio utterance batch size {utterance_audio.size(0)} != {batch_size}"

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
        # 由于我们使用了batch_first=True，所以需要调整输入格式
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, 
                         self.utt_shared_t, self.utt_shared_v, self.utt_shared_a), dim=1)
        h = self.transformer_encoder(h)
        h = torch.cat((h[:, 0], h[:, 1], h[:, 2], h[:, 3], h[:, 4], h[:, 5]), dim=1)
        o = self.fusion(h)
        return o

    def reconstruct(self,):
        # 由于现在共享空间已经经过了时空解耦和对齐融合，我们直接使用这些处理后的特征
        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)

    def shared_private(self, utterance_t, utterance_v, utterance_a):
        # 确保所有输入特征的批次大小一致
        batch_size = utterance_t.size(0)
        
        # 确保所有特征的批次大小一致
        assert utterance_v.size(0) == batch_size, f"Visual batch size {utterance_v.size(0)} != text batch size {batch_size}"
        assert utterance_a.size(0) == batch_size, f"Audio batch size {utterance_a.size(0)} != text batch size {batch_size}"
        
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
        
        # 确保共享表示的维度和批次大小正确
        hidden_dim = self.config.hidden_size
        
        # 确保共享表示的形状是 [batch_size, hidden_dim]
        shared_t = shared_t.view(batch_size, hidden_dim)
        shared_v = shared_v.view(batch_size, hidden_dim)
        shared_a = shared_a.view(batch_size, hidden_dim)
        
        # 将向量扩展为序列形式 [batch_size, 1, hidden_dim]
        shared_t_seq = shared_t.view(batch_size, 1, hidden_dim)
        shared_v_seq = shared_v.view(batch_size, 1, hidden_dim)
        shared_a_seq = shared_a.view(batch_size, 1, hidden_dim)
        
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
        
        # 对每个模态的时间和空间特征进行对齐融合
        t_fused = self.alignment_fusion(t_temporal, t_spatial).squeeze(1)  # [batch_size, hidden_dim]
        v_fused = self.alignment_fusion(v_temporal, v_spatial).squeeze(1)
        a_fused = self.alignment_fusion(a_temporal, a_spatial).squeeze(1)
        
        # 将融合后的特征作为最终的共享表示
        self.utt_shared_t = t_fused
        self.utt_shared_v = v_fused
        self.utt_shared_a = a_fused

    def forward(self, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        o = self.alignment(sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask)
        return o
