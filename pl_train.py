import os 
import torch 
import torch.nn as nn
import pytorch_lightning as pl

# from aim import Run
from aim.pytorch_lightning import AimLogger

from dataset import MAEDataset
import models_mae


BATCH_SIZE = 20
EPOCHS = 100
RANDOM_INIT = True
EPOCHS = 500
DEVICE = 'cpu'
continue_from_checkpoint = False
DEVICE = 'cpu'
learning_rate = 1e-4
l1 = 1
intersection_threshold = 0.1
# torch.cuda.empty_cache()


# aim_logger = AimLogger(experiment='wt_background_random_init')
# aim_logger.log_hyperparams({
#     "max_epochs": 10,
#     # 'learning_rate': learning_rate,
#     # 'negative_loss_coef': l1,
#     # 'class_intersection_threshold':intersection_threshold,
#     # 'batch_size':BATCH_SIZE,
#     # 'random_init':RANDOM_INIT
# })

# Run['hparams'] = {
#     'learning_rate': learning_rate,
#     'negative_loss_coef': l1,
#     'class_intersection_threshold':intersection_threshold,
#     'batch_size':BATCH_SIZE,
#     'random_init':RANDOM_INIT}

def cosine_distance_torch(x1, x2=None):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t())


class ContrastiveLoss(nn.Module):
    def __init__(self, num_classes=5, margin=1.0) -> None:
        super().__init__()
        self.num_classes=num_classes
        self.margin = margin

    def forward(self, img_enc_1, labels, img_enc_2=None):
        if not img_enc_2:
            cos_dist = cosine_distance_torch(img_enc_1)
        else:
            cos_dist = cosine_distance_torch(img_enc_1, img_enc_2)

        # d = 0 means y1 and y2 are supposed to be same
        # d = 1 means y1 and y2 are supposed to be different

        distance_matrix = (labels.repeat(labels.shape[0], 1) - labels.repeat(labels.shape[0], 1).T)
        distance_matrix = distance_matrix.abs().sign()

        positive_loss = (1 - distance_matrix) * cos_dist
        positive_loss /= (1 - distance_matrix).sum()
        positive_loss = torch.nan_to_num(positive_loss)

        delta = self.margin - cos_dist # if margin == 1, then 1 - cos_dist == cos_sim
        delta= torch.clamp(delta, min=0.0, max=None)
        negative_loss = distance_matrix * delta
        negative_loss /= (distance_matrix).sum()
        negative_loss = torch.nan_to_num(negative_loss)

        agg_loss = torch.zeros((self.num_classes+1, self.num_classes+1))
        agg_d = torch.zeros((self.num_classes+1, self.num_classes+1))
        label_masks = [labels==i for i in range(self.num_classes+1)]
        for i in range(self.num_classes + 1):
            for j in range(self.num_classes + 1):
                agg_loss[i][j] = cos_dist[label_masks[i]][:, label_masks[j]].mean()
                agg_d[i][j] = distance_matrix[label_masks[i]][:, label_masks[j]].mean()

        print(*[x.sum().item() for x in label_masks])
        print(agg_loss)
        # print(agg_d)

        return positive_loss.sum(), negative_loss.sum()


class LightningMAE(pl.LightningModule):
    def __init__(self, model, l1=0.5, lr=1e-4, num_classes=5, margin=1):
        super().__init__()
        self.model_mae = model
        self.l1 = l1
        self.lr = lr
        self.min_loss = 10
        self.criterion = ContrastiveLoss(num_classes=num_classes, margin=margin)
        # self.logger = AimLogger(experiment='wt_background_random_init')
        # self.logger.log_hyperparams({
        #     "max_epochs": 10,
        #     # 'learning_rate': learning_rate,
        #     # 'negative_loss_coef': l1,
        #     # 'class_intersection_threshold':intersection_threshold,
        #     # 'batch_size':BATCH_SIZE,
        #     # 'random_init':RANDOM_INIT
        # })
        # self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # torch.cuda.empty_cache()
        # print("Memory allocation init:", torch.cuda.memory_allocated(DEVICE) / 1024 / 1024 / 1024)
        # print("Max memory allocation init:", torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024 / 1024)
        img = torch.einsum('nhwc->nchw', batch['image'])
        # print("Getting image embeddings...")
        img_enc, mask, _ = self.model_mae.forward_encoder(img.float(), mask_ratio=0)
        img_enc = img_enc[:, 1:, :].reshape(-1, img_enc.shape[-1])
        # print("Memory allocation after embeddings:", torch.cuda.memory_allocated(DEVICE) / 1024 / 1024 / 1024)
        # print("Max memory allocation after embeddings:", torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024 / 1024)
        # print("Loss counting...")
        loss = self.criterion(img_enc, batch['indices_labels'].reshape(-1))
        total_loss = loss[0] + self.l1 * loss[1]
        self.log('train_loss', total_loss)
        # print("Memory allocation after loss:", torch.cuda.memory_allocated(DEVICE) / 1024 / 1024 / 1024)
        # print("Max memory allocation after loss:", torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024 / 1024)
        print()
        # print(f'Iter: {batch_idx}, pos_loss: {loss[0].item()}, neg_loss = {self.l1} * {loss[1].item()}, loss: {total_loss.item()}')

        if self.min_loss > total_loss:
            self.min_loss = total_loss.item()

            torch.save(self.model_mae.state_dict(), "/mnt/2tb/alla/mae/mae_contastive/background_random_init/best_model.pth")


        return total_loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model_mae.parameters(), lr=self.lr)
        return optimizer


    def forward(self, img):
        encoded_image, mask, indices = self.model_mae.forward_encoder(img, mask_ratio=0)
        return encoded_image[:, 1:, :]

if __name__ == '__main__':

    #### init dataset ####

    root = '/mnt/2tb/hrant/FAIR1M/fair1m_1000/train1000/'
    path_ann = os.path.join(root, 'few_shot_8.json')
    path_imgs = os.path.join(root, 'images')
    dataset = MAEDataset(path_ann, path_imgs, intersection_threshold=intersection_threshold, resize_image=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    #### init model ####

    arch='mae_vit_large_patch16'
    model_mae = getattr(models_mae, arch)()
    if continue_from_checkpoint:

        chkpt_dir = '/mnt/2tb/hrant/checkpoints/mae_models/mae_visualize_vit_large_ganloss.pth'
        checkpoint = torch.load(chkpt_dir, map_location=DEVICE)
        msg = model_mae.load_state_dict(checkpoint['model'], strict=False)

        chkpt_dir = '/mnt/2tb/alla/mae/mae_contastive/background/lightning_logs/version_1/checkpoints/epoch=142-step=143.ckpt'
        model_mae = LightningMAE.load_from_checkpoint(chkpt_dir, model=model_mae)
        model_mae = model_mae.model_mae


    else:
        # chkpt_dir = '/mnt/2tb/hrant/checkpoints/mae_models/mae_visualize_vit_large.pth'
        chkpt_dir = '/mnt/2tb/hrant/checkpoints/mae_models/mae_visualize_vit_large_ganloss.pth'
        checkpoint = torch.load(chkpt_dir, map_location=DEVICE)
        if not RANDOM_INIT:
            msg = model_mae.load_state_dict(checkpoint['model'], strict=False)
        


    model = LightningMAE(model_mae, lr=learning_rate, l1=l1)
    trainer = pl.Trainer(logger=True, enable_checkpointing=True, limit_predict_batches=BATCH_SIZE, max_epochs=EPOCHS, log_every_n_steps=1, \
        default_root_dir="/mnt/2tb/alla/mae/mae_contastive/background_random_init", accelerator=DEVICE,) # devices=1,) # gpus=-1)
    # trainer = pl.Trainer(logger=AimLogger(experiment='experiment_name'), accelerator=DEVICE, devices=1,)
    trainer.fit(model=model, train_dataloaders=dataloader)

