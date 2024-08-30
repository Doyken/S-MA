"""
Name: dev_process
Date: 2024/6/28 上午10:26
Version: 1.0
"""

import torch
# from transformers import AdamW
from torch.optim import Adam, AdamW, SGD
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from util.write_file import WriteFile
import dev_process
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import ModelParam,MaskedKLDivLoss,MaskedNLLLoss
import torch.nn.modules as nn
# import tensorflow as tf



def train_process(opt, train_loader, dev_loader,test_loader, cl_model, critertion, log_summary_writer:SummaryWriter=None,  gamma_1=1.0, gamma_2=1.0,gamma_3=1.0,tokenizer=None, image_id_list=None):
    optimizer = None

    pre_train_model_param = [name for name, param in cl_model.named_parameters() if 'text_model' in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in cl_model.named_parameters() if n in pre_train_model_param],
            "lr":0,
        },
        {
            "params": [p for n, p in cl_model.named_parameters() if n not in pre_train_model_param],
            "lr": opt.fuse_lr,
        },
    ]

    if opt.optim == 'adam':
        optimizer = Adam(optimizer_grouped_parameters, betas=(opt.optim_b1, opt.optim_b2))
    elif opt.optim == 'adamw':
        optimizer = AdamW(optimizer_grouped_parameters, betas=(opt.optim_b1, opt.optim_b2))
    elif opt.optim == 'sgd':
        optimizer = SGD(optimizer_grouped_parameters, momentum=opt.momentum)

    orgin_param = ModelParam()
    augment_param = ModelParam()

    last_F1 = 0
    last_Accuracy = 0
    for epoch in trange(opt.epoch, desc='Epoch:'):
        y_true = []
        y_pre = []
        run_loss = 0
        total_labels = 0
        new_epoch= epoch

        cl_model.train()
        cl_model.zero_grad()

        if epoch >= opt.train_fuse_model_epoch:
            optimizer.param_groups[0]['lr'] = opt.lr
            optimizer.param_groups[1]['lr'] = opt.lr

        train_loader_tqdm = tqdm(train_loader, desc='Train Iteration:')
        epoch_step_num = epoch * train_loader_tqdm.total
        step_num = 0
        for index, data in enumerate(train_loader_tqdm):
            texts_origin, bert_attention_mask, image_origin, text_image_mask, labels,\
                texts_augment, bert_attention_mask_augment, image_augment, text_image_mask_augment, target_labels = data

            if opt.cuda is True:
                texts_origin = texts_origin.cuda()
                bert_attention_mask = bert_attention_mask.cuda()
                image_origin = image_origin.cuda()
                text_image_mask = text_image_mask.cuda()
                labels = labels.cuda()
                texts_augment = texts_augment.cuda()
                bert_attention_mask_augment = bert_attention_mask_augment.cuda()
                image_augment = image_augment.cuda()
                text_image_mask_augment = text_image_mask_augment.cuda()
                for i in range(len(target_labels)):
                    target_labels[i] = target_labels[i].cuda()

            orgin_param.set_data_param(texts=texts_origin, bert_attention_mask=bert_attention_mask, images=image_origin, text_image_mask=text_image_mask,)
            augment_param.set_data_param(texts=texts_augment, bert_attention_mask=bert_attention_mask_augment, images=image_augment, text_image_mask=text_image_mask_augment)
            kl_loss = MaskedKLDivLoss()
            loss_function = MaskedNLLLoss()

            train_test =opt.train_test
            t_log_prob, i_log_prob, all_log_prob, all_prob, kl_t_log_prob, kl_a_log_prob, kl_all_prob= cl_model(orgin_param,index,new_epoch,train_test)


            loss = gamma_1 * loss_function(all_log_prob, labels) + \
                   gamma_1 * (loss_function(t_log_prob, labels) + loss_function(i_log_prob, labels)) + \
                   gamma_1 * (kl_loss(kl_t_log_prob, kl_all_prob, ) + kl_loss(kl_a_log_prob, kl_all_prob, ))

            loss = (loss) / opt.acc_batch_size



            loss.backward()
            train_loader_tqdm.set_description("Train Iteration, loss: %.6f, lr: %e" %
                                              (loss, optimizer.param_groups[0]['lr']))

            if (index + 1) % opt.acc_grad == 0:
                if log_summary_writer:
                    log_summary_writer.add_scalar('train_info/loss', loss.item(), global_step=step_num + epoch_step_num)
                    log_summary_writer.add_scalar('train_info/classify_loss', loss.item(), global_step=step_num + epoch_step_num)
                    log_summary_writer.add_scalar('train_info/fuse_lr', optimizer.param_groups[1]['lr'], global_step=step_num + epoch_step_num)
                optimizer.step()
                optimizer.zero_grad()
            step_num += 1


            _, predicted = torch.max(all_prob, 1)
            y_true.extend(labels.cpu())
            y_pre.extend(predicted.cpu())
            run_loss += loss.item()
            total_labels += labels.size(0)

        run_loss /= total_labels
        y_true = np.array(y_true)
        y_pre = np.array(y_pre)
        train_accuracy = accuracy_score(y_true, y_pre)
        train_F1_weighted = f1_score(y_true, y_pre, average='weighted')
        train_R_weighted = recall_score(y_true, y_pre, average='weighted')
        train_precision_weighted = precision_score(y_true, y_pre, average='weighted')
        train_F1 = f1_score(y_true, y_pre, average='macro')
        train_R = recall_score(y_true, y_pre, average='macro')
        train_precision = precision_score(y_true, y_pre, average='macro')

        save_content = 'Epoch: %d:\nTrain: Accuracy: %.6f, F1(weighted): %.6f, Precision(weighted): %.6f, R(weighted): %.6f, F1(macro): %.6f, Precision: %.6f, R: %.6f, loss: %.6f' % \
                       (epoch, train_accuracy, train_F1_weighted, train_precision_weighted, train_R_weighted, train_F1, train_precision, train_R, run_loss)
        WriteFile(opt.save_model_path, 'train_correct_log.txt', save_content + '\n', 'a+')
        print(save_content, ' ' * 200)

        if log_summary_writer:
            log_summary_writer.add_scalar('train_info/loss_epoch', run_loss, global_step=epoch)
            log_summary_writer.add_scalar('train_info/acc', train_accuracy, global_step=epoch)
            log_summary_writer.add_scalar('train_info/f1_w', train_F1_weighted, global_step=epoch)
            log_summary_writer.add_scalar('train_info/r_w', train_R_weighted, global_step=epoch)
            log_summary_writer.add_scalar('train_info/p_w', train_precision_weighted, global_step=epoch)
            log_summary_writer.add_scalar('train_info/f1_ma', train_F1, global_step=epoch)
            log_summary_writer.flush()

        train_log = {
            "epoch": epoch,
            "train_accuracy": train_accuracy,
            "train_F1": train_F1,
            "train_R": train_R,
            "train_precision": train_precision,
            "train_F1_weighted": train_F1_weighted,
            "train_precision_weighted": train_precision_weighted,
            "train_R_weighted": train_R_weighted,
            "run_loss": run_loss
        }
        last_F1, last_Accuracy = dev_process.dev_process(opt,critertion, cl_model, dev_loader, epoch, test_loader, last_F1, last_Accuracy, train_log, log_summary_writer)
