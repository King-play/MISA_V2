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
    """Finding the mean along dim"""
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)

def masked_max(tensor, mask, dim):
    """Finding the max along dim"""
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf
    return (masked + neg_inf).max(dim=dim)



# let's define a simple model that can deal with multimodal variable length sequence
class MISA(nn.Module):
    def __init__(self, config):
        super(MISA, self).__init__()

        self.config = config
        self.text_size = config.embedding_size
        
        # 根据配置选择视觉特征处理方式
        if config.use_facet_visual:
            self.visual_size = config.facet_visual_size  # 35维Facet特征
        else:
            self.visual_size = config.visual_size  # 原始视觉特征
            
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
        
        # 视觉特征处理：根据配置选择不同的处理方式
        if not config.use_facet_visual:
            # 原始方式：使用LSTM处理视觉特征
            self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
            self.vrnn2 = rnn(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
            self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        else:
            # 新方式：直接处理Facet特征，不使用LSTM
            self.visual_fc = nn.Sequential(
                nn.Linear(self.visual_size, config.hidden_size),
                self.activation,
                nn.LayerNorm(config.hidden_size)
            )
        
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

        # 视觉投影层：根据特征处理方式调整
        if config.use_facet_visual:
            # Facet特征已经在visual_fc中处理过了，这里只需要恒等映射
            self.project_v = nn.Sequential()
            self.project_v.add_module('project_v', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
            self.project_v.add_module('project_v_activation', self.activation)
            self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))
        else:
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
        # semi-shared encoders
        ##########################################
        # Text-Visual semi-shared encoder
        self.semi_shared_tv = nn.Sequential()
        self.semi_shared_tv.add_module('semi_shared_tv_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.semi_shared_tv.add_module('semi_shared_tv_activation_1', nn.Sigmoid())
        
        # Text-Audio semi-shared encoder  
        self.semi_shared_ta = nn.Sequential()
        self.semi_shared_ta.add_module('semi_shared_ta_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.semi_shared_ta.add_module('semi_shared_ta_activation_1', nn.Sigmoid())
        
        # Visual-Audio semi-shared encoder
        self.semi_shared_va = nn.Sequential()
        self.semi_shared_va.add_module('semi_shared_va_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.semi_shared_va.add_module('semi_shared_va_activation_1', nn.Sigmoid())

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

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

        ##########################################
        # semi-shared discriminators
        ##########################################
        self.semi_discriminator_tv = nn.Sequential()
        self.semi_discriminator_tv.add_module('semi_discriminator_tv_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.semi_discriminator_tv.add_module('semi_discriminator_tv_activation', self.activation)
        self.semi_discriminator_tv.add_module('semi_discriminator_tv_2', nn.Linear(in_features=config.hidden_size, out_features=3))

        self.semi_discriminator_ta = nn.Sequential()
        self.semi_discriminator_ta.add_module('semi_discriminator_ta_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.semi_discriminator_ta.add_module('semi_discriminator_ta_activation', self.activation)
        self.semi_discriminator_ta.add_module('semi_discriminator_ta_2', nn.Linear(in_features=config.hidden_size, out_features=3))

        self.semi_discriminator_va = nn.Sequential()
        self.semi_discriminator_va.add_module('semi_discriminator_va_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.semi_discriminator_va.add_module('semi_discriminator_va_activation', self.activation)
        self.semi_discriminator_va.add_module('semi_discriminator_va_2', nn.Linear(in_features=config.hidden_size, out_features=3))

        # 融合层
        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size*12, out_features=self.config.hidden_size*4))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.config.hidden_size*4, out_features= output_size))

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        
    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        lengths = lengths.cpu() 
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

        # 视觉特征处理：根据配置选择不同方式
        if self.config.use_facet_visual:
            # 使用Facet特征：需要正确处理维度
            #print(f"Visual shape: {visual.shape}")  # 调试信息
            #print(f"Lengths shape: {lengths.shape}")  # 调试信息

            if len(visual.shape) == 3:
                # visual shape: [seq_len, batch_size, feature_dim] 或 [batch_size, seq_len, feature_dim]
                # 需要确定visual的维度顺序
                
                if visual.shape[0] == batch_size:
                    # [batch_size, seq_len, feature_dim]
                    seq_len = visual.shape[1]
                    feature_dim = visual.shape[2]
                    
                    # 创建mask: [batch_size, seq_len, 1]
                    mask = (torch.arange(seq_len).unsqueeze(0).to(lengths.device) < lengths.unsqueeze(1)).float().unsqueeze(2)
                    
                    # 计算加权平均
                    utterance_video = (visual * mask).sum(dim=1) / mask.sum(dim=1)
                    
                else:
                    # [seq_len, batch_size, feature_dim] - 需要转置
                    visual = visual.permute(1, 0, 2)  # 转为 [batch_size, seq_len, feature_dim]
                    seq_len = visual.shape[1]
                    feature_dim = visual.shape[2]
                    
                    # 创建mask: [batch_size, seq_len, 1]
                    mask = (torch.arange(seq_len).unsqueeze(0).to(lengths.device) < lengths.unsqueeze(1)).float().unsqueeze(2)
                    
                    # 计算加权平均
                    utterance_video = (visual * mask).sum(dim=1) / mask.sum(dim=1)
                    
            elif len(visual.shape) == 2:
                # visual shape: [batch_size, feature_dim] - 已经是聚合后的特征
                utterance_video = visual
            else:
                raise ValueError(f"Unexpected visual shape: {visual.shape}")
            
            # 通过全连接层处理
            utterance_video = self.visual_fc(utterance_video)
            
        else:
            # 原始方式：使用LSTM提取视觉特征
            final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
            utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # extract features from acoustic modality (保持不变)
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

        # 半公共空间判别器
        reversed_semi_tv_t = ReverseLayerF.apply(self.utt_semi_shared_tv_t, self.config.reverse_grad_weight)
        reversed_semi_tv_v = ReverseLayerF.apply(self.utt_semi_shared_tv_v, self.config.reverse_grad_weight)
        reversed_semi_tv_a = ReverseLayerF.apply(self.utt_semi_shared_ta_a, self.config.reverse_grad_weight)

        self.semi_domain_label_tv_t = self.semi_discriminator_tv(reversed_semi_tv_t)
        self.semi_domain_label_tv_v = self.semi_discriminator_tv(reversed_semi_tv_v)
        self.semi_domain_label_tv_a = self.semi_discriminator_tv(reversed_semi_tv_a)

        reversed_semi_ta_t = ReverseLayerF.apply(self.utt_semi_shared_ta_t, self.config.reverse_grad_weight)
        reversed_semi_ta_a = ReverseLayerF.apply(self.utt_semi_shared_ta_a, self.config.reverse_grad_weight)
        reversed_semi_ta_v = ReverseLayerF.apply(self.utt_semi_shared_tv_v, self.config.reverse_grad_weight)

        self.semi_domain_label_ta_t = self.semi_discriminator_ta(reversed_semi_ta_t)
        self.semi_domain_label_ta_a = self.semi_discriminator_ta(reversed_semi_ta_a)
        self.semi_domain_label_ta_v = self.semi_discriminator_ta(reversed_semi_ta_v)

        reversed_semi_va_v = ReverseLayerF.apply(self.utt_semi_shared_va_v, self.config.reverse_grad_weight)
        reversed_semi_va_a = ReverseLayerF.apply(self.utt_semi_shared_va_a, self.config.reverse_grad_weight)
        reversed_semi_va_t = ReverseLayerF.apply(self.utt_semi_shared_ta_t, self.config.reverse_grad_weight)

        self.semi_domain_label_va_v = self.semi_discriminator_va(reversed_semi_va_v)
        self.semi_domain_label_va_a = self.semi_discriminator_va(reversed_semi_va_a)
        self.semi_domain_label_va_t = self.semi_discriminator_va(reversed_semi_va_t)

        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator( (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a)/3.0 )
        
        # For reconstruction
        self.reconstruct()
        
        # Transformer融合
        h = torch.stack((
            self.utt_private_t, self.utt_private_v, self.utt_private_a,
            self.utt_semi_shared_tv_t, self.utt_semi_shared_tv_v,
            self.utt_semi_shared_ta_t, self.utt_semi_shared_ta_a,
            self.utt_semi_shared_va_v, self.utt_semi_shared_va_a,
            self.utt_shared_t, self.utt_shared_v, self.utt_shared_a
        ), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8], h[9], h[10], h[11]), dim=1)
        o = self.fusion(h)
        return o
    
    def reconstruct(self,):
        self.utt_t = (self.utt_private_t + self.utt_semi_shared_tv_t + self.utt_semi_shared_ta_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_semi_shared_tv_v + self.utt_semi_shared_va_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_semi_shared_ta_a + self.utt_semi_shared_va_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)

    def shared_private(self, utterance_t, utterance_v, utterance_a):
        
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)

        # Private components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        # Semi-shared components
        self.utt_semi_shared_tv_t = self.semi_shared_tv(utterance_t)
        self.utt_semi_shared_tv_v = self.semi_shared_tv(utterance_v)
        
        self.utt_semi_shared_ta_t = self.semi_shared_ta(utterance_t)
        self.utt_semi_shared_ta_a = self.semi_shared_ta(utterance_a)
        
        self.utt_semi_shared_va_v = self.semi_shared_va(utterance_v)
        self.utt_semi_shared_va_a = self.semi_shared_va(utterance_a)

        # Fully shared components
        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)

    def forward(self, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        batch_size = lengths.size(0)
        o = self.alignment(sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask)
        return o
