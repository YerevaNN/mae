import numpy as np
import torch
import os
import torch.nn as nn
from torch.optim import Adam
import torchmetrics

from memory_profiler import profile

from dataset import MAEDataset
import models_mae


# imagenet_mean = np.array([0.485, 0.456, 0.406])
# imagenet_std = np.array([0.229, 0.224, 0.225])
arch='mae_vit_large_patch16'
chkpt_dir = '/mnt/2tb/hrant/checkpoints/mae_models/mae_visualize_vit_large.pth'
chkpt_dir = 'best_model.pth'
# chkpt_dir = "model_final.pth"


class ContrastiveLoss(nn.Module):
    def __init__(self, l1=0.5, m=1.0, num_classes=5):
        super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
        self.l1 = l1
        self.margin = m  # margin or radius
        self.num_classes = num_classes

    def forward(self, y1, y2, labels):
        right = labels.clone()
        right[(labels==0).nonzero()] = 20
        d = (labels.repeat(labels.shape[0], 1) - right.repeat(right.shape[0], 1).T).abs().sign()
        bg_mask = 1 - torch.outer((labels==0).int(), (labels==0).int())
        print(f"y's: {y1.device}, labels: {labels.device}, d: {d.device}")
        # torch.save(d, 'd_matr.pth')
        # torch.save(labels, 'labels_for_d_mart.pth')
        # print(d)
        # d = 0 means y1 and y2 are supposed to be same
        # d = 1 means y1 and y2 are supposed to be different
        
        cos_dist = 1 - torchmetrics.functional.pairwise_cosine_similarity(y1, y2)
        # print(torch.unique((1-d).reshape(-1), return_counts=True))
        loss_pos = (1-d) * cos_dist
        loss_pos = loss_pos / (1-d).sum()
        # print('mult pos:', torch.unique(mult_pos.reshape(-1), return_counts=True))

        delta = self.margin - cos_dist
        delta = torch.clamp(delta, min=0.0, max=None)
        loss_neg = d * delta 
        loss_neg = loss_neg * bg_mask
        loss_neg = loss_neg / (d * bg_mask).sum()

        loss = loss_pos + loss_neg
        
        agg_loss = torch.zeros((self.num_classes+1, self.num_classes+1)).to(loss.device)
        agg_d = torch.zeros((self.num_classes+1, self.num_classes+1)).to(loss.device)
        label_masks = [labels==i for i in range(self.num_classes+1)]
        for i in range(self.num_classes + 1):
            for j in range(self.num_classes + 1):
                print(cos_dist.shape, label_masks[i].shape)
                agg_loss[i][j] = cos_dist[label_masks[i]][:, label_masks[j]].mean()
                agg_d[i][j] = d[label_masks[i]][:, label_masks[j]].mean()
        print([x.sum() for x in label_masks])
        print(agg_loss)
        print(agg_d)
        
        return loss_pos.sum(), loss_neg.sum()
        # loss_matrix = mult_pos + self.l1 * mult_neg
        
        # return loss_matrix.mean()


model_mae = getattr(models_mae, arch)()
checkpoint = torch.load(chkpt_dir, map_location='cuda')
# msg = model_mae.load_state_dict(checkpoint['model'], strict=False)
msg = model_mae.load_state_dict(checkpoint, strict=False)
device = 'cuda'
model_mae.cuda()

l1 = 1

root = '/mnt/2tb/hrant/FAIR1M/fair1m_1000/train1000/'
path_ann = os.path.join(root, 'few_shot_8.json')
path_imgs = os.path.join(root, 'images')
dataset = MAEDataset(path_ann, path_imgs, resize_image=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
optimizer = Adam(model_mae.parameters(), lr=1e-4)
criterion = ContrastiveLoss(l1=l1, num_classes=5)

num_epochs = 1
min_loss = torch.tensor(5)
for epoch in range(num_epochs):
    model_mae.train()
    # loss = torch.tensor(0)
    for i, ds in enumerate(data_loader):
        optimizer.zero_grad()

        img = torch.einsum('nhwc->nchw', ds['image']).to(device)
        print("image device:", img.device)
        img_enc, mask, _ = model_mae.forward_encoder(img.float(), mask_ratio=0)
        img_enc = img_enc[:, 1:, :].reshape(-1, img_enc.shape[-1])
        print("img_enc device:", img_enc.device)
        loss = criterion(img_enc, img_enc, ds['indices_labels'].reshape(-1).to(device))
        total_loss = (loss[0] + l1*loss[1])
        total_loss.backward()
        optimizer.step()

        if i % 3 == 0:
            print("Epoch: {}/{}, Iteration: {}/{}, Loss: {:.4f} ".format(epoch+1, num_epochs, i+1, len(data_loader), (total_loss).item()))
            print(f"pos: {loss[0].item():.4f}, neg: {l1} * {loss[1].item():.4f}")
            torch.save(model_mae.state_dict(), 'model_current.pth')
            with open('loss_log.txt', 'a') as f:
                f.write("Epoch: {}/{}, Iteration: {}/{}, Loss: {:.4f}\n".format(epoch+1, num_epochs, i+1, len(data_loader), (total_loss).item()))

    # if min_loss > (loss[0] + loss[1]).item():
    #     torch.save(model_mae.state_dict(), 'best_model.pth')
    #     min_loss = (loss[0]+loss[1]).item()

# Save the trained model
# torch.save(model_mae.state_dict(), 'model_final.pth')


