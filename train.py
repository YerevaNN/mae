import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torchmetrics

from dataset import MAEDataset
import models_mae


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
arch='mae_vit_large_patch16'
chkpt_dir = '/mnt/2tb/hrant/checkpoints/mae_models/mae_visualize_vit_large.pth'


class ContrastiveLoss(nn.Module):
    def __init__(self, m=1.0):
        super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
        self.margin = m  # margin or radius

    def forward(self, y1, y2, labels):
        d = (labels.repeat(labels.shape[0], 1) - labels.repeat(labels.shape[0], 1).T).abs().sign()

        # d = 0 means y1 and y2 are supposed to be same
        # d = 1 means y1 and y2 are supposed to be different
        
        cos_dist = torchmetrics.functional.pairwise_cosine_similarity(y1, y2, zero_diagonal=True)
        mult_pos = torch.matmul((1-d).double(), cos_dist.double())

        delta = self.margin - cos_dist
        delta = torch.clamp(delta, min=0.0, max=None)
        mult_neg = torch.matmul(d.double(), delta.double())
        
        loss_matrix = mult_pos + mult_neg
        
        return loss_matrix.mean()


model_mae = getattr(models_mae, arch)()
checkpoint = torch.load(chkpt_dir, map_location='cuda')
msg = model_mae.load_state_dict(checkpoint['model'], strict=False)

dataset = MAEDataset('./annotations/few_shot_8_fair1m.json', '/mnt/2tb/hrant/FAIR1M/fair1m_1000/train1000/images', resize_image=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
optimizer = Adam(model_mae.parameters())
criterion = ContrastiveLoss()

num_epochs = 1

for epoch in range(num_epochs):
    model_mae.train()
    for i, ds in enumerate(data_loader):
        optimizer.zero_grad()

        img = torch.einsum('nhwc->nchw', ds['image'])
        img_enc, mask, _ = model_mae.forward_encoder(img.float(), mask_ratio=0)
        img_enc = img_enc[:, 1:, :].reshape(-1, img_enc.shape[-1])
        loss = criterion(img_enc, img_enc, ds['indices_labels'].reshape(-1))
        loss.backward()
        optimizer.step()

        if i % 40 == 0:
            print("Epoch: {}/{}, Iteration: {}/{}, Loss: {:.4f}".format(epoch+1, num_epochs, i+1, len(data_loader), loss.item()))
            # torch.save(model_mae.state_dict(), 'model_{}.pth'.format(epoch))
# Save the trained model
torch.save(model_mae.state_dict(), 'model_.pth')
