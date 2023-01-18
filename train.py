import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from dataset import MAEDataset
import models_mae


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
arch='mae_vit_large_patch16'
chkpt_dir = 'mae_visualize_vit_large.pth'


class ContrastiveLoss(nn.Module):
    def __init__(self, labels, m=2.0):
        super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
        self.margin = m  # margin or radius

    def forward(self, y1, y2, labels):
        d = (labels.repeat(labels.shape[0], 1) - labels.repeat(labels.shape[0], 1).T).abs().sign()

        # d = 0 means y1 and y2 are supposed to be same
        # d = 1 means y1 and y2 are supposed to be different

        cos_dist = nn.functional.pairwise_cosine_similarity(y1, y2, zero_diagonal=True)
        mult_pos = torch.matmul((1-d).double(), cos_dist.double())
        
        delta = self.margin - cos_dist
        delta = torch.clamp(delta, min=0.0, max=None)
        mult_neg = torch.matmul(d.double(), delta.double())
        
        return mult_pos + mult_neg


model_mae = getattr(models_mae, arch)()
checkpoint = torch.load(chkpt_dir, map_location='cpu')
msg = model_mae.load_state_dict(checkpoint['model'], strict=False)

dataset = MAEDataset('./annotations/few_shot_8_fair1m.json', resize_image=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
optimizer = Adam(model_mae.parameters())
criterion = ContrastiveLoss()

num_epochs = 100

for epoch in range(num_epochs):
    for i, (img, _, _, _, _, label, _) in enumerate(data_loader):
        optimizer.zero_grad()
        x = x.unsqueeze(dim=0)
        img_enc, mask, _ = models_mae(img.permute(2, 0, 1).unsqueeze(dim=0))
        # output1, mask, _ = model_mae(img_enc[1:], img_enc[1:])
        loss = criterion(img_enc[1:], img_enc[1:], label)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("Epoch: {}/{}, Iteration: {}/{}, Loss: {:.4f}".format(epoch+1, num_epochs, i+1, len(data_loader), loss.item()))
            torch.save(model_mae.state_dict(), 'model_{}.pth'.format(epoch))
# Save the trained model
# torch.save(model_mae.state_dict(), 'model_.pth')