import torch
import numpy as np
from neural_networks import multiModalRepresentation_diff, multiModalRepresentation, encoderDecoder
from dataloader import gestureBlobDataset, gestureBlobBatchDataset, gestureBlobMultiDataset, size_collate_fn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import os
from datetime import datetime
from typing import Tuple, List, Union
import sys
import math

# Use subset sampler for train test split

def calc_labels(y: torch.Tensor) -> torch.Tensor:
    # out = torch.ones(y.size()[0], 15, dtype = torch.float32)*(0.01/14)
    out = torch.zeros(y.size()[0], 15, dtype = torch.long)
    #print(out.size())
    for i in range(out.size()[0]):
        # out[i, int(y[i].item()) - 1] = 0.99
        out[i, int(y[i].item()) - 1] = 1
    return(out)

def train_multimodal_embeddings(lr: float, num_epochs: int, blobs_folder_path:str, weights_save_path: str, weight_decay: float) -> None:
    if torch.cuda.is_available():
        print('Using CUDA')

    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)

    gesture_dataset = gestureBlobDataset(blobs_folder_path = blobs_folder_path)
    dataset = gestureBlobBatchDataset(gesture_dataset = gesture_dataset, random_tensor = 'random')
    dataloader = DataLoader(dataset = dataset, batch_size = 2, shuffle = True)

    loss_function = torch.nn.MSELoss()
    # loss_function = torch.nn.KLDivLoss()
    # net = multiModalRepresentation_diff(out_features = 2048, lstm_num_layers= 2, parser = 'cnn')
    net = multiModalRepresentation(out_features = 512, lstm_num_layers= 2, parser = 'cnn')
    net = net.train()
    if torch.cuda.is_available():
        net.cuda()

    # Remove comment to initialized at a pretrained point
    # net.load_state_dict(torch.load(os.path.join(weights_save_path,'two_stream_Knot_Tying_2020-02-01_20:16:16.pth' )))

    optimizer = torch.optim.Adam(params = net.parameters(), lr = lr, weight_decay = weight_decay)

    for epoch in range(num_epochs):
        running_loss = 0
        count = 0
        print('Epoch {}'.format(epoch + 1))
        # for idx in range(len(dataset)):
        for data in dataloader:
            current_tensor, random_tensor, y_match, y_rand = data
            curr_opt, curr_kin = current_tensor
            _, rand_kin = random_tensor
            # kin = torch.cat([kinematics, kinematics_rand], dim = 1)
            if torch.cuda.is_available():
                curr_opt = curr_opt.cuda()
                curr_kin = curr_kin.cuda()
                y_match = y_match.cuda()
                rand_kin = rand_kin.cuda()
                y_rand = y_rand.cuda()
            optimizer.zero_grad()
            # import pdb; pdb.set_trace()
            out1 = net((curr_opt, curr_kin))
            out2 = net((curr_opt, rand_kin))
            loss = loss_function(out2, y_rand) + loss_function(out1, y_match)
            # loss = loss_function(out2.log(), y_rand) + loss_function(out1.log(), y_match) # Use for KLDivLoss
            loss.backward()
            optimizer.step()
            print('Out1: {}'.format(out1[0, :]))
            print('Out2: {}'.format(out2[0, :]))
            print('Current loss2 = {}'.format(loss.item()))
            running_loss += loss.item()
            count += 1
        print(out1[0, :,])
        print(out2[0, :])
        print('\n Epoch: {}, Loss: {}'.format(epoch + 1, running_loss/count))

    print('Finished training.')
    print('Saving state dict.')

    now = datetime.now()
    now = '_'.join((str(now).split('.')[0]).split(' '))
    file_name = 'multimodal_' + dataset_name + '_' + now + '.pth'
    file_name = os.path.join(weights_save_path, file_name)
    torch.save(net.state_dict(), file_name)

    print('State dict saved at timestamp {}'.format(now))

def train_encoder_decoder_embeddings(net: encoderDecoder, lr: float, num_epochs: int, blobs_folder_path: str, weights_save_path: str, weight_decay: float, dataset_name: str, model_dim: int) -> None:
    if torch.cuda.is_available():
        print('Using CUDA')
    print('embedding size:', model_dim)

    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)

    gesture_dataset = gestureBlobDataset(blobs_folder_path = blobs_folder_path)
    dataloader = DataLoader(dataset = gesture_dataset, batch_size = 32, shuffle = False, collate_fn = size_collate_fn)

    loss_function = torch.nn.MSELoss()
    # loss_function = torch.nn.KLDivLoss()
    # net = encoderDecoder(embedding_dim = model_dim)
    net = net.train()
    if torch.cuda.is_available():
        net.cuda()

    # Remove comment to initialized at a pretrained point
    # net.load_state_dict(torch.load(os.path.join(weights_save_path,'two_stream_Knot_Tying_2020-02-01_20:16:16.pth' )))

    optimizer = torch.optim.Adam(params = net.parameters(), lr = lr, weight_decay = weight_decay)

    for epoch in range(num_epochs):
        running_loss = 0
        count = 0
        print('Epoch {}'.format(epoch + 1))
        for data in dataloader:
            curr_opt, curr_kin = data
            if torch.cuda.is_available():
                curr_opt = curr_opt.cuda()
                curr_kin = curr_kin.cuda()
            optimizer.zero_grad()
            out1 = net(curr_opt)
            loss = loss_function(out1, curr_kin)
            loss.backward()
            optimizer.step()
            # print('Current loss = {}'.format(loss.item()))
            running_loss += loss.item()
            count += 1
        print('\n Epoch: {}, Loss: {}'.format(epoch + 1, running_loss/count))

    print('Finished training.')
    print('Saving state dict.')

    now = datetime.now()
    now = '_'.join((str(now).split('.')[0]).split(' '))
    file_name = 'multimodal_' + dataset_name + '_' + now + '.pth'
    file_name = os.path.join(weights_save_path, file_name)
    torch.save(net.state_dict(), file_name)

    print('State dict saved at timestamp {}'.format(now))

def train_encoder_decoder_embeddings_moco(net: encoderDecoder, lr: float, num_epochs: int, blobs_folder_path: str, weights_save_path: str, weight_decay: float, dataset_name: str, model_dim: int) -> None:
    if torch.cuda.is_available():
        print('Using CUDA')
    print('embedding size:', model_dim)

    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)

    gesture_dataset = gestureBlobDataset(blobs_folder_path = blobs_folder_path)
    dataloader = DataLoader(dataset = gesture_dataset, batch_size = 64, shuffle = False, collate_fn = size_collate_fn)

    # loss_function = torch.nn.KLDivLoss()
    kin_loss = torch.nn.MSELoss()
    contrastive_loss = torch.nn.CrossEntropyLoss()

    # net = encoderDecoder(embedding_dim = model_dim)
    net = net.train()
    if torch.cuda.is_available():
        net.cuda()

    # Remove comment to initialized at a pretrained point
    # net.load_state_dict(torch.load(os.path.join(weights_save_path,'two_stream_Knot_Tying_2020-02-01_20:16:16.pth' )))

    optimizer = torch.optim.Adam(params = net.parameters(), lr = lr, weight_decay = weight_decay)

    for epoch in range(num_epochs):
        running_loss = 0
        count = 0
        print('Epoch {}'.format(epoch + 1))
        for data in dataloader:
            curr_opt, curr_kin = data
            if torch.cuda.is_available():
                curr_opt = curr_opt.cuda()
                curr_kin = curr_kin.cuda()
            optimizer.zero_grad()

            noise_mult = 1e-6
            curr_opt_aug = curr_opt + noise_mult * torch.randn(*curr_opt.shape).cuda()

            output, target = net(im_q = curr_opt_aug, im_k = curr_opt)
            loss = contrastive_loss(output, target) + kin_loss(target, curr_kin)

            loss.backward()
            optimizer.step()
            # print('Current loss = {}'.format(loss.item()))
            running_loss += loss.item()
            count += 1
        print('\n Epoch: {}, Loss: {}'.format(epoch + 1, running_loss/count))

    print('Finished training.')
    print('Saving state dict.')

    now = datetime.now()
    now = '_'.join((str(now).split('.')[0]).split(' '))
    file_name = 'multimodal_' + dataset_name + '_' + now + '.pth'
    file_name = os.path.join(weights_save_path, file_name)
    torch.save(net.state_dict(), file_name)

    print('State dict saved at timestamp {}'.format(now))

def train_encoder_decoder_embeddings_contrastive(net: encoderDecoder, lr: float, num_epochs: int, blobs_folder_path: str, weights_save_path: str, weight_decay: float, dataset_name: str, model_dim: int, noise_variance: float) -> None:
    if torch.cuda.is_available():
        print('Using CUDA')
    print('embedding size:', model_dim)
    print('noise variance:', noise_variance)

    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)

    gesture_dataset = gestureBlobDataset(blobs_folder_path = blobs_folder_path)
    dataloader = DataLoader(dataset = gesture_dataset, batch_size = 48, shuffle = True, collate_fn = size_collate_fn)

    kin_loss = torch.nn.MSELoss()
    #con_loss = torch.nn.KLDivLoss()
    exp_loss = lambda x: torch.sum(x**2) / x.shape[0]
    con_loss = torch.nn.MSELoss()

    # net = encoderDecoder(embedding_dim = model_dim)
    net = net.train()
    if torch.cuda.is_available():
        net.cuda()

    optimizer = torch.optim.Adam(params = net.parameters(), lr = lr, weight_decay = weight_decay)

    for epoch in range(num_epochs):
        running_loss = 0
        count = 0
        print('Epoch {}'.format(epoch + 1))
        for data in dataloader:
            opt, kin = data
            if torch.cuda.is_available():
                opt = opt.cuda()
                kin = kin.cuda()
            optimizer.zero_grad()

            opt_aug = opt + noise_variance * torch.randn(*opt.shape).cuda()

            embedding, predict = net(opt)
            embedding_aug, predict_aug = net(opt_aug)

            batch_size = opt.shape[0]
            con_perm = torch.Tensor([(i, x) for i, x in enumerate(torch.randperm(batch_size)) if i != x]).T.long()

            l_kin = kin_loss(predict, kin) + kin_loss(predict_aug, kin) # kinematic loss
            l_exp = exp_loss(embedding) # explosive loss (keep vectors small magnitude)
            l_con = con_loss(embedding, embedding_aug) - con_loss(embedding[con_perm[0]], embedding[con_perm[1]]) # contrastive loss
            loss = l_kin + l_exp + l_con

            loss.backward()
            optimizer.step()
            # print('Current loss = {}'.format(loss.item()))
            running_loss += loss.item()
            count += 1
        print('\n Epoch: {}, Loss: {}'.format(epoch + 1, running_loss/count))

    print('Finished training.')
    print('Saving state dict.')

    now = datetime.now()
    now = '_'.join((str(now).split('.')[0]).split(' '))
    file_name = 'multimodal_' + dataset_name + '_' + now + '.pth'
    file_name = os.path.join(weights_save_path, file_name)
    torch.save(net.state_dict(), file_name)

    print('State dict saved at timestamp {}'.format(now))

def train_encoder_decoder_embeddings_contrastive_kin(net: encoderDecoder, lr: float, num_epochs: int, blobs_folder_path: str, weights_save_path: str, weight_decay: float, dataset_name: str, model_dim: int) -> None:
    if torch.cuda.is_available():
        print('Using CUDA')
    print('embedding size:', model_dim)
    print('epochs:', num_epochs)
    print('kin contrastive mode')

    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)

    gesture_dataset = gestureBlobDataset(blobs_folder_path = blobs_folder_path)
    dataloader = DataLoader(dataset = gesture_dataset, batch_size = 48, shuffle = True, collate_fn = size_collate_fn)

    kin_loss = torch.nn.MSELoss()
    #con_loss = torch.nn.KLDivLoss()
    exp_loss = lambda x: torch.sum(x**2) / x.shape[0]
    con_loss = torch.nn.MSELoss()

    # net = encoderDecoder(embedding_dim = model_dim)
    net = net.train()
    if torch.cuda.is_available():
        net.cuda()

    optimizer = torch.optim.Adam(params = net.parameters(), lr = lr, weight_decay = weight_decay)

    for epoch in range(num_epochs):
        running_loss = 0
        count = 0
        print('Epoch {}'.format(epoch + 1))
        for data in dataloader:
            opt, kin = data
            if torch.cuda.is_available():
                opt = opt.cuda()
                kin = kin.cuda()
            optimizer.zero_grad()

            embedding, predict = net(opt)

            batch_size = opt.shape[0]
            match = []
            unmatch = []
            for i in range(batch_size):
                diffs = torch.sum((kin - kin[i])**2, axis=[1,2,3])
                assert diffs.shape == (batch_size,), kin.shape
                unmatch_idx = torch.argmax(diffs)
                diffs[i] = math.inf
                match_idx = torch.argmin(diffs)
                assert unmatch_idx != i and match_idx != i, (i, match_idx, unmatch_idx, diffs)
                match.append(match_idx)
                unmatch.append(unmatch_idx)
            match = torch.Tensor(match).long()
            unmatch = torch.Tensor(unmatch).long()

            l_kin = kin_loss(predict, kin) # kinematic loss
            l_exp = exp_loss(embedding) # explosive loss (keep vectors small magnitude)
            l_con = con_loss(embedding, embedding[match]) - con_loss(embedding, embedding[unmatch]) # contrastive loss
            loss = l_kin + l_exp + l_con

            loss.backward()
            optimizer.step()
            # print('Current loss = {}'.format(loss.item()))
            running_loss += loss.item()
            count += 1
        print('\n Epoch: {}, Loss: {}'.format(epoch + 1, running_loss/count))

    print('Finished training.')
    print('Saving state dict.')

    now = datetime.now()
    now = '_'.join((str(now).split('.')[0]).split(' '))
    file_name = 'multimodal_' + dataset_name + '_' + now + '.pth'
    file_name = os.path.join(weights_save_path, file_name)
    torch.save(net.state_dict(), file_name)

    print('State dict saved at timestamp {}'.format(now))

def train_encoder_decoder_multidata_embeddings(lr: float, num_epochs: int, blobs_folder_paths_list: List[str], weights_save_path: str, weight_decay: float) -> None:
    if torch.cuda.is_available():
        print('Using CUDA')

    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)

    gesture_dataset = gestureBlobMultiDataset(blobs_folder_paths_list = blobs_folder_paths_list)
    dataloader = DataLoader(dataset = gesture_dataset, batch_size = 128, shuffle = False, collate_fn = size_collate_fn)

    loss_function = torch.nn.MSELoss()
    # loss_function = torch.nn.KLDivLoss()
    net = encoderDecoder(embedding_dim = 2048)
    net = net.train()
    if torch.cuda.is_available():
        net.cuda()

    # Remove comment to initialized at a pretrained point
    # net.load_state_dict(torch.load(os.path.join(weights_save_path,'two_stream_Knot_Tying_2020-02-01_20:16:16.pth' )))

    optimizer = torch.optim.Adam(params = net.parameters(), lr = lr, weight_decay = weight_decay)

    for epoch in range(num_epochs):
        running_loss = 0
        count = 0
        print('Epoch {}'.format(epoch + 1))
        for data in dataloader:
            curr_opt, curr_kin = data
            if torch.cuda.is_available():
                curr_opt = curr_opt.cuda()
                curr_kin = curr_kin.cuda()
            optimizer.zero_grad()
            out1 = net(curr_opt)
            loss = loss_function(out1, curr_kin)
            loss.backward()
            optimizer.step()
            # print('Current loss = {}'.format(loss.item()))
            running_loss += loss.item()
            count += 1
        print('\n Epoch: {}, Loss: {}'.format(epoch + 1, running_loss/count))

    print('Finished training.')
    print('Saving state dict.')

    now = datetime.now()
    now = '_'.join((str(now).split('.')[0]).split(' '))
    file_name = 'multimodal_multidata_' + now + '.pth'
    file_name = os.path.join(weights_save_path, file_name)
    torch.save(net.state_dict(), file_name)

    print('State dict saved at timestamp {}'.format(now))


def main():
    blobs_folder_path = '../jigsaw_dataset/Knot_Tying/blobs'
    lr = 1e-3
    num_epochs = 1000
    weights_save_path = './weights_save'
    weight_decay = 1e-8
    dataset_name = 'Knot_Tying'

    blobs_folder_paths_list = ['../jigsaw_dataset/Knot_Tying/blobs', '../jigsaw_dataset/Needle_Passing/blobs', '../jigsaw_dataset/Suturing/blobs']

    # train_multimodal_embeddings(lr = lr, num_epochs = num_epochs, blobs_folder_path = blobs_folder_path, weights_save_path = weights_save_path, weight_decay = weight_decay)
    # train_encoder_decoder_embeddings(lr = lr, num_epochs = num_epochs, blobs_folder_path = blobs_folder_path, weights_save_path = weights_save_path, weight_decay = weight_decay, dataset_name = dataset_name)
    train_encoder_decoder_multidata_embeddings(lr = lr, num_epochs = num_epochs, blobs_folder_paths_list = blobs_folder_paths_list, weights_save_path = weights_save_path, weight_decay = weight_decay)

if __name__ == '__main__':
    main()
