import os
import numpy as np
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.tensorboard

def log_gpu_memory(logger):
    device = torch.device('cuda:0')
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert bytes to MB
    cached = torch.cuda.memory_reserved(device) / (1024 ** 2)  # Convert bytes to MB
    logger.info(f"GPU memory allocated: {allocated:.2f} MB")
    logger.info(f"GPU memory cached: {cached:.2f} MB")

class P2V(nn.Module):
    def __init__(self, n_products, size, device):
        super().__init__()
        self.device = device
        # trainable variables
        self.wi = torch.nn.Embedding(n_products, size, sparse=True).to(device)
        with torch.no_grad():
            self.wi.weight.uniform_(-0.025, 0.025)
        self.wo = torch.nn.Embedding(n_products, size, sparse=True).to(device)
        with torch.no_grad():
            self.wo.weight.uniform_(-0.025, 0.025)

    def forward(self, center, context, negative_samples):

        # embed products (center, context, negative_samples)
        wi_center = self.wi(center)
        wo_positive_samples = self.wo(context)
        wo_negative_samples = self.wo(negative_samples)

        # logits
        logits_positive_samples = torch.einsum(
            "ij,ij->i", (wi_center, wo_positive_samples)
        )
        logits_negative_samples = torch.einsum(
            "ik,ijk->ij", (wi_center, wo_negative_samples)
        )

        # loss
        loss_positive_samples = F.binary_cross_entropy_with_logits(
            input=logits_positive_samples,
            target=torch.ones_like(logits_positive_samples),
            reduction="sum",
        )
        loss_negative_samples = F.binary_cross_entropy_with_logits(
            input=logits_negative_samples,
            target=torch.zeros_like(logits_negative_samples),
            reduction="sum",
        )

        n_samples = logits_positive_samples.shape[0] * (
            logits_negative_samples.shape[1] + 1
        )
        return (loss_positive_samples + loss_negative_samples) / n_samples


class TrainerP2V:
    def __init__(self, model, train, validation, path, device, n_batch_log=500):
        self.model = model.to(device)
        self.train = train
        self.validation = validation
        self.optimizer = torch.optim.SparseAdam(params=list(model.parameters()))
        self.path = path
        self.device = device
        os.makedirs(f"{self.path}/weights", exist_ok=True)
        self.writer_train = torch.utils.tensorboard.SummaryWriter(f"{self.path}/train")
        self.writer_val = torch.utils.tensorboard.SummaryWriter(f"{self.path}/val")
        self.n_batch_log = n_batch_log
        self.global_batch = 0
        self.epoch = 0
        self.batch = 0

    def fit(self, epochs):

        for _ in range(epochs):
            print(f"epoch = {self.epoch}")

            for ce, co, ns in self.train:
                ce, co, ns = ce.to(self.device), co.to(self.device), ns.to(self.device)
                self.batch += 1
                self.global_batch += 1
                self.optimizer.zero_grad()
                loss_train = self.model(ce, co, ns)
                loss_train.backward()
                self.optimizer.step()
                self.writer_train.add_scalar("loss", loss_train, self.global_batch)

                if self.batch % self.n_batch_log == 1:
                    self._callback_batch()
                    log_gpu_memory(logger)

            self.epoch += 1

        self.writer_train.flush()
        self.writer_train.close()
        self.writer_val.flush()
        self.writer_val.close()

    def _callback_batch(self):
        # validation loss
        self.model.eval()
        with torch.no_grad():
            list_loss_validation = []
            for ce, co, ns in self.validation:
                ce, co, ns = ce.to(self.device), co.to(self.device), ns.to(self.device)
                list_loss_validation.append(self.model(ce, co, ns).item())
            loss_validation = np.mean(list_loss_validation)
        self.writer_val.add_scalar("loss", loss_validation, self.global_batch)
        self.model.train()

        # save input embedding
        np.save(
            f"{self.path}/weights/wi_{self.epoch:02d}_{self.batch:06d}.npy",
            self.get_wi(),
        )

    def predict(self):
        None

    def get_wi(self):
        return self.model.wi.weight.cpu().detach().numpy()

    def get_wo(self):
        return self.model.wo.weight.cpu().detach().numpy()