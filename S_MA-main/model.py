"""
Name: dev_process
Date: 2024/6/28 上午10:26
Version: 1.0
"""

import numpy as np
import torch.nn.modules as nn
import torchvision.models as cv_models
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import BertConfig, BertForPreTraining, RobertaForMaskedLM, RobertaModel, RobertaConfig, AlbertModel, AlbertConfig
import math
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from pre_model import RobertaEncoder
import copy
from sklearn.manifold import TSNE


class MaskedKLDivLoss(nn.Module):
    def __init__(self):
        super(MaskedKLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction='sum')

    def forward(self, log_pred, target):

        loss = self.loss(log_pred, target)
        return loss


class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target):
        # mask_ = mask.view(-1, 1)
        # print("target",target.shape)
        # print("pred",pred.shape)
        # print("mask",mask.shape)
        if type(self.weight) == type(None):
            loss = self.loss(pred , target)
        else:
            loss = self.loss(pred, target) \
                   / torch.sum(self.weight[target])
        return loss


class ModelParam:
    def __init__(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token

    def set_data_param(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token


def get_extended_attention_mask(attention_mask, input_shape):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with athe same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


class ActivateFun(nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)


class TextModel(nn.Module):
    def __init__(self, opt):
        super(TextModel, self).__init__()
        abl_path = ''

        if opt.text_model == 'bert-base':
            self.config = BertConfig.from_pretrained(abl_path + 'bert-base-uncased/')
            self.model = BertForPreTraining.from_pretrained(abl_path + 'bert-base-uncased/', config=self.config)
            self.model = self.model.bert

        for param in self.model.parameters():
            param.requires_grad = True

        self.output_dim = self.model.encoder.layer[11].output.dense.out_features

    def get_output_dim(self):
        return self.output_dim

    def get_config(self):
        return self.config

    def get_encoder(self):
        model_encoder = copy.deepcopy(self.model.encoder)
        return model_encoder

    def forward(self, input, attention_mask):
        output = self.model(input, attention_mask=attention_mask)
        return output.last_hidden_state, output.pooler_output



class ImageModel(nn.Module):
    def __init__(self, opt):
        super(ImageModel, self).__init__()
        if opt.image_model == 'resnet-152':
            self.resnet = cv_models.resnet152(pretrained=True)
        elif opt.image_model == 'resnet-101':
            self.resnet = cv_models.resnet101(pretrained=True)
        elif opt.image_model == 'resnet-50':
            self.resnet = cv_models.resnet50(pretrained=True)
        elif opt.image_model == 'resnet-34':
            self.resnet = cv_models.resnet34(pretrained=True)
        elif opt.image_model == 'resnet-18':
            self.resnet = cv_models.resnet18(pretrained=True)
        self.resnet_encoder = nn.Sequential(*(list(self.resnet.children())[:-2]))
        self.resnet_avgpool = nn.Sequential(list(self.resnet.children())[-2])
        self.output_dim = self.resnet_encoder[7][2].conv3.out_channels

        for param in self.resnet.parameters():
            if opt.fixed_image_model:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def get_output_dim(self):
        return self.output_dim

    def forward(self, images):
        image_encoder = self.resnet_encoder(images)
        # image_encoder = self.conv_output(image_encoder)
        image_cls = self.resnet_avgpool(image_encoder)
        image_cls = torch.flatten(image_cls, 1)
        return image_encoder, image_cls


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))



class PositionwiseFeedForward(nn.Module):
    # yaigai
    def __init__(self, d_model, d_ff, dropout=0.3):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.3):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2). \
            contiguous().view(batch_size, -1, head_count * dim_per_head)
        output = self.linear(context)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb
        return x

class linear(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(linear, self).__init__()
        self.linear = nn.Linear(d_model,768)
    def forward(self, x):
        return self.linear(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout,):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b):
        if inputs_a.equal(inputs_b):
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            context = self.self_attn(inputs_b, inputs_b, inputs_b)
            out = self.dropout(context) + inputs_b
        else:
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b



            context = self.self_attn(inputs_a, inputs_a, inputs_b)


            out = self.dropout(context) + inputs_a



        return self.feed_forward(out)
class Multimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(Multimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b):
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)
        utters = torch.cat([a_new, b_new], dim=-2)
        utters_fc = torch.cat([self.fc(a).unsqueeze(-2), self.fc(b).unsqueeze(-2)], dim=-2)
        utters_softmax = self.softmax(utters_fc)
        utters_three_model = utters_softmax * utters
        final_rep = torch.sum(utters_three_model, dim=-2, keepdim=False)
        return final_rep

class TransformerEncoder(nn.Module):
    def __init__(self,opt, d_model, d_ff, heads, layers, dropout=0.1,):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(opt, d_model, heads, d_ff, dropout)
             for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b):
        if x_a.equal(x_b):
            x_b = self.pos_emb(x_b)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_b, x_b)
        else:
            x_a = self.pos_emb(x_a)
            x_a = self.dropout(x_a)
            x_b = self.pos_emb(x_b)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_a, x_b)
        return x_b



class Transformer_Based_Model(nn.Module):
    def __init__(self, opt, temp=1, n_head=8,
                 n_classes=3, hidden_dim=768, n_speakers=0, dropout=0.2, image_output_type="all"):
        super(Transformer_Based_Model, self).__init__()
        self.temp = temp
        self.n_classes = n_classes
        self.n_speakers = n_speakers
        self.image_output_type = image_output_type

        # 加的
        self.text_model = TextModel(opt)
        self.image_model = ImageModel(opt)
        self.fuse_type =  opt.fuse_type
        self.tran_dim= opt.tran_dim

        self.image_config = copy.deepcopy(self.text_model.get_config())
        self.text_config = copy.deepcopy(self.text_model.get_config())

        self.image_config.num_attention_heads = opt.tran_dim // 64
        self.image_config.hidden_size = opt.tran_dim
        self.image_config.num_hidden_layers = opt.image_num_layers
        self.image_encoder = RobertaEncoder(self.image_config)

        self.text_change = nn.Sequential(
            nn.Linear(self.text_model.get_output_dim(), self.tran_dim),
            ActivateFun(opt)
        )
        self.image_change = nn.Sequential(
            nn.Linear(self.image_model.get_output_dim(), self.tran_dim),
            ActivateFun(opt)
        )
        self.image_cls_change = nn.Sequential(
            nn.Linear(self.image_model.get_output_dim(), self.tran_dim),
            ActivateFun(opt)
        )
        self.features_reduce_t = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_a = nn.Linear(3 * hidden_dim, hidden_dim)

        # Intra- and Inter-modal Transformers
        self.t_t = TransformerEncoder(opt, d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout,)
        self.i_t = TransformerEncoder(opt, d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.i_i = TransformerEncoder(opt, d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t_i = TransformerEncoder(opt, d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)






        self.features_reduce_t = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_a = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_v = nn.Linear(3 * hidden_dim, hidden_dim)

        # Multimodal-level Gated Fusion
        self.last_gate = Multimodal_GatedFusion(hidden_dim)

        # Emotion Classifier
        self.t_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        self.a_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        self.v_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1, n_classes)
        )

        self.all_output_layer = nn.Linear(hidden_dim, n_classes)


        self.hostory = Hostroy_Mul()


    def forward(self, text_inputs, image_inputs, text_image_mask,index,epoch,attention_mask=None):
        last_hidden_state, pooler_output = self.text_model(
            text_inputs, attention_mask=attention_mask, )


        textf = self.text_change(last_hidden_state)
        icouf, image_cls = self.image_model(image_inputs)
        if self.image_output_type == 'all':
            image_encoder = icouf.contiguous().view(icouf.size(0), -1,
                                                                 icouf.size(1))
            image_encoder_init = self.image_change(image_encoder)
            image_cls_init = self.image_cls_change(image_cls)
            icouf = torch.cat((image_cls_init.unsqueeze(1), image_encoder_init), dim=1)
        else:
            image_cls_init = self.image_cls_change(image_cls)
            icouf= image_cls_init.unsqueeze(1)

        image_mask = text_image_mask[:, -icouf.size(1):]
        extended_attention_mask = get_extended_attention_mask(image_mask, icouf.size())
        image_init = self.image_encoder(icouf,
                                        attention_mask=None,
                                        head_mask=None,
                                        encoder_hidden_states=None,
                                        encoder_attention_mask=extended_attention_mask,
                                        past_key_values=None,
                                        use_cache=False,
                                        output_attentions=self.text_config.output_attentions,
                                        output_hidden_states=(self.text_config.output_hidden_states),
                                        return_dict=self.text_config.use_return_dict
                                        )
        icouf = image_init.last_hidden_state

        # Intra- and Inter-modal Transformers
        t_t_transformer_out = self.t_t(textf, textf)
        i_t_transformer_out = self.i_t(icouf, textf)


        i_i_transformer_out = self.i_i(icouf, icouf)
        t_i_transformer_out = self.t_i(textf, icouf)



        t_transformer_out = (torch.cat([t_t_transformer_out, i_t_transformer_out], dim=1))
        i_transformer_out = (torch.cat([i_i_transformer_out, t_i_transformer_out], dim=1))




        all_transformer_out = self.last_gate(t_transformer_out, i_transformer_out)


        # # Emotion Classifier
        text_image_length = t_transformer_out.size(2)
        t_transformer_out = torch.sum(t_transformer_out, dim=1) / text_image_length
        text_image_length = i_transformer_out.size(2)
        i_transformer_out = torch.sum(i_transformer_out, dim=1) / text_image_length
        # 相识度计算




        if self.fuse_type == 'max':
            all_transformer_out = torch.max(all_transformer_out, dim=1)[0]
        elif self.fuse_type == 'ave':
            text_image_length = all_transformer_out.size(2)
            all_transformer_out = torch.sum(all_transformer_out, dim=1) / text_image_length
        else:
            raise Exception('fuse_type设定错误')

        t_final_out = self.t_output_layer(t_transformer_out)
        i_final_out = self.a_output_layer(i_transformer_out)

        all_final_out = self.all_output_layer(all_transformer_out)
        t_log_prob = F.log_softmax(t_final_out, 1)
        i_log_prob = F.log_softmax(i_final_out, 1)
        all_log_prob = F.log_softmax(all_final_out, 1)


        all_prob = F.softmax(all_final_out, 1)


        kl_t_log_prob = F.log_softmax(t_final_out / self.temp, 1)
        kl_a_log_prob = F.log_softmax(i_final_out / self.temp, 1)
        kl_all_prob = F.softmax(all_final_out / self.temp, 1)




        return t_log_prob, i_log_prob, all_log_prob, all_prob, kl_t_log_prob, kl_a_log_prob, kl_all_prob,

class CLModel(nn.Module):
    def __init__(self, opt):
        super(CLModel, self).__init__()
        self.Transformer_Based_Model =Transformer_Based_Model(opt)
        self.output_classify = nn.Sequential(
            nn.Dropout(opt.l_dropout),
            nn.Linear(768, 768 // 2),
            ActivateFun(opt),
            nn.Linear(768 // 2, 3)
        )

    def forward(self, data_orgin: ModelParam, index, epoch, data_augment: ModelParam = None, labels=None, target_labels=None):
        t_log_prob, i_log_prob, all_log_prob, all_prob, kl_t_log_prob, kl_a_log_prob, kl_all_prob, = self.Transformer_Based_Model(text_inputs=data_orgin.texts,image_inputs=data_orgin.images,
                                                                                                    text_image_mask=data_orgin.text_image_mask,index=index, epoch=epoch)

        return t_log_prob, i_log_prob, all_log_prob, all_prob, kl_t_log_prob, kl_a_log_prob, kl_all_prob

class TensorBoardModel(nn.Module):
    def __init__(self, opt):
        super(TensorBoardModel, self).__init__()
        self.cl_model = CLModel(opt)

    def forward(self, texts, bert_attention_mask, images,text_image_mask, index, epoch,train_test,
                texts_augment, bert_attention_mask_augment, images_augment, text_image_mask_augment, label):
        orgin_param = ModelParam()
        augment_param = ModelParam()
        orgin_param.set_data_param(texts=texts, bert_attention_mask=bert_attention_mask, images=images, text_image_mask=text_image_mask)
        augment_param.set_data_param(texts=texts_augment, bert_attention_mask=bert_attention_mask_augment, images=images_augment, text_image_mask=text_image_mask_augment)
        return self.cl_model(orgin_param, augment_param, index, epoch, label, [torch.ones(1, dtype=torch.int64) for _ in range(3)])
