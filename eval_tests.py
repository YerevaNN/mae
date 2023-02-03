import os
import cv2
import json
import torch
from tqdm import tqdm
from dataset import MAEDataset
import matplotlib.pyplot as plt
from pl_train import LightningMAE
from torchmetrics.functional import pairwise_cosine_similarity as cos_dist

import models_mae

# device = 'cuda:0'
device = 'cpu'

root = '/mnt/2tb/hrant/FAIR1M/fair1m_1000/train1000/'
path_ann = os.path.join(root, 'few_shot_8.json')
path_imgs = os.path.join(root, 'images')

dataset = MAEDataset(path_ann, path_imgs, resize_image=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)

root_val = '/mnt/2tb/hrant/FAIR1M/fair1m_1000/val1000/'
path_ann = os.path.join(root, 'few_shot_8.json')
path_imgs = os.path.join(root, 'images')

dataset_val = MAEDataset(path_ann, path_imgs, resize_image=True)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=5, shuffle=True)
#### init model ####

# chkpt_dir = '/mnt/2tb/alla/mae/mae_contastive/best_model_check.pth'
chkpt_dir = '/mnt/2tb/alla/mae/mae_contastive/lightning_logs/version_1/checkpoints/epoch=54-step=55.ckpt'
chkpt_dir_old = '/mnt/2tb/hrant/checkpoints/mae_models/mae_visualize_vit_large.pth'
checkpoint = torch.load(chkpt_dir, map_location='cuda')
arch = 'mae_vit_large_patch16'
model_mae = getattr(models_mae, arch)()
checkpoint_old = torch.load(chkpt_dir_old, map_location='cuda')
print(checkpoint.keys())
msg = model_mae.load_state_dict(checkpoint_old['model'], strict=False)


print(msg)
# model_mae = LightningMAE(model=model_mae)
model_mae = LightningMAE.load_from_checkpoint(chkpt_dir, model=model_mae)

model_mae.eval()
model_mae.to(device)


def count_reference(dataloader, model, num_classes=5):
    # model.eval()
    reference_sum = torch.zeros((num_classes+1, 1024)).to(device)
    counts = torch.zeros(num_classes+1).to(device)
    for i, ds in tqdm(enumerate(dataloader), total=len(dataloader)):
        img = torch.einsum('nhwc->nchw', ds['image']).to(device)
        # img_enc, _, _ = model.forward_encoder(img.float(), mask_ratio=0)
        # img_enc = img_enc[:, 1:, :].reshape(-1, img_enc.shape[-1])
        img_enc = model(img.float())
        img_enc = img_enc.reshape(-1, img_enc.shape[-1])
        index_labels = ds['indices_labels'].reshape(-1).to(device)
        for j in torch.unique(torch.tensor(index_labels, dtype=int).clone().detach()):
            indices = (index_labels == j).nonzero()
            counts[j] += indices.shape[0]
            new = img_enc[indices].reshape(-1, img_enc.shape[-1])
            # print('class:', j, 'sum:', new.shape, 'shape:', new.sum(dim=0).shape)
            reference_sum[j] += new.sum(dim=0).clone().detach()
            
    return reference_sum, counts


def return_img_embed(dataloader, model):
    # model.eval()
    output = []
    images = []
    labels = []
    for i, ds in tqdm(enumerate(dataloader), total=len(dataloader)):
        img = torch.einsum('nhwc->nchw', ds['image'])
        # img_enc, _, _ = model.forward_encoder(img.float(), mask_ratio=0)
        # img_enc = img_enc[:, 1:, :] #.reshape(-1, img_enc.shape[-1])
        img_enc = model(img.float()) #.reshape(-1)
        index_labels = ds['indices_labels'] #.reshape(-1)
        labels.append(index_labels)
        images.append(img.detach())
        output.append(img_enc.detach())

            
    output = torch.cat(output, dim=0)
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    return (output, images, labels)


def return_embed_per_class(dataloader, model, num_class=1):
    # model.eval()
    output = []
    images = []
    for i, ds in tqdm(enumerate(dataloader), total=len(dataloader)):
        img = torch.einsum('nhwc->nchw', ds['image']).to(device)
        img_enc = model(img.float()).reshape(-1, img_enc.shape[-1])
        index_labels = ds['indices_labels'].reshape(-1).to(device)
        indices = (index_labels == num_class).nonzero()
        if indices.shape[0] != 0:
            images.append(img.detach())
            new = img_enc[indices].reshape(-1, img_enc.shape[-1])
            output.append(new.detach())

    output = torch.cat(output, dim=0)
    images = torch.cat(images, dim=0)
    return (output, images)


def count_tp_fp_fn(embeds, ref_mean, labels, num_classes=5):
    tp, fn = torch.zeros(num_classes+1), torch.zeros(num_classes+1)
    fp, tn = torch.zeros(num_classes+1), torch.zeros(num_classes+1)
    true_preds = torch.zeros(num_classes+1)
    print(embeds.shape, ref_mean.shape)
    embeds_flat = cos_dist(embeds.reshape(-1, embeds.shape[-1]), ref_mean).argmax(dim=1)
    labels_flat = labels.reshape(-1)
    for cls in range(num_classes+1):
        tp[cls] = (embeds_flat[(labels_flat==cls).nonzero()] == cls).sum()
        fp[cls] = (embeds_flat[(labels_flat==cls).nonzero()] != cls).sum()
        fn[cls] = (embeds_flat[(labels_flat!=cls).nonzero()] == cls).sum()
        true_preds[cls] = ((labels_flat==cls).int() * (embeds_flat==cls).int()).sum()
    return tp, fp, fn, true_preds


def count_accuracy(num_classes=5):
    true_preds = torch.zeros(num_classes+1)
    for cls in range(num_classes+1):
        true_preds[cls] += ((labels[0]==cls).int() * (cos_sim==cls).int()).sum()
    pass

ref_sum, counts = count_reference(dataloader, model_mae)
# val_sum, val_count = count_reference(dataloader_val, model_mae)

ref_mean = (ref_sum.T / counts).T

# val_mean, ref_mean = (val_sum.T / val_count).T, (ref_sum.T / counts).T


embeds, imgs, labels = return_img_embed(dataloader_val, model_mae)

tp, fp, fn, true_preds = count_tp_fp_fn(embeds, ref_mean, labels)

precision = tp / (tp + fp)
recall = tp / (tp + fn)

print(f'model precision is: {precision}, and recall is {recall}, true_predictions {true_preds/(tp + fp)}')