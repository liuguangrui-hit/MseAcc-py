import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from catalyst import dl
from catalyst.contrib.nn.modules import Flatten, GlobalMaxPool2d, Lambda
from catalyst.callbacks.optimizer import OptimizerCallback
import numpy as np
import inspect
import matplotlib.pyplot as plt
import pandas as pd
from GAN.MLP import MLP
from GAN.ResFc import ResFc


class GANDataset(Dataset):
    def __init__(self, input_file,target_file):
        self.input_items = np.load(input_file)
        self.target_items = np.load(target_file)[[0,10000]]


    def __len__(self):
        return min(self.input_items.shape[0],self.target_items.shape[0])

    def __getitem__(self, idx):
        return self.input_items[idx, :],self.target_items[idx,:]


class CustomRunner(dl.Runner):
    def handle_batch(self, batch):
        input, target= batch[0], batch[1]
        batch_size = input.shape[0]
        latent_dim=input.shape[1]
        random_latent_vectors = 0.1*torch.randn(batch_size, latent_dim).cuda()+0.9*input

        generated_attacks = self.model["generator"](
            random_latent_vectors).detach()

        combined_attacks = torch.cat([generated_attacks, target])

        labels = torch.cat([torch.ones((batch_size, 1)),
                            torch.zeros((batch_size, 1))]).cuda()
        labels += 0.05 * torch.rand(labels.shape).cuda()

        combined_predictions = self.model["discriminator"](combined_attacks)

        misleading_labels = torch.zeros((batch_size, 1)).cuda()
        generated_attacks = self.model["generator"](random_latent_vectors)
        generated_predictions = self.model["discriminator"](generated_attacks)

        self.batch = {
            "combined_predictions": combined_predictions,
            "labels": labels,
            "generated_predictions": generated_predictions,
            "misleading_labels": misleading_labels,
        }


if __name__ == "__main__":
    discriminator = MLP(layers=[("fc", 78, 128), ("lrelu",),
                                ("fc", 128, 128), ("lrelu"), ("fc", 128, 1), ("s")])
    # discriminator.load_state_dict(torch.load("discriminator.pt"))
    generator=ResFc(78,78)

    model = {"generator": generator, "discriminator": discriminator}
    criterion = {"generator": nn.BCEWithLogitsLoss(
    ), "discriminator": nn.BCEWithLogitsLoss()}
    optimizer = {
        "generator": torch.optim.Adam(generator.parameters(), lr=1e-6, betas=(0.5, 0.999)),
        "discriminator": torch.optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999)),
    }
    loaders = {"train": DataLoader(GANDataset("../nids_models/data/time1_fail_DNN.npy", "../nids_models/data/time1_success_DNN.npy"), batch_size=32, shuffle=True)
               }

    runner = CustomRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,

        callbacks=[
            dl.CriterionCallback(
                input_key="combined_predictions",
                target_key="labels",
                metric_key="loss_discriminator",
                criterion_key="discriminator",
            ),
            dl.CriterionCallback(
                input_key="generated_predictions",
                target_key="misleading_labels",
                metric_key="loss_generator",
                criterion_key="generator",
            ),
            OptimizerCallback("loss_generator",
                              model_key="generator", optimizer_key="generator"),
            # OptimizerCallback(
            #     "loss_discriminator", model_key="discriminator", optimizer_key="discriminator"),
        ],
        valid_loader="train",
        valid_metric="loss_discriminator",
        minimize_valid_metric=False,
        num_epochs=10,
        verbose=True,
        logdir="logs_gan/",
    )
    torch.save(model["generator"].state_dict(), "g10_time1_generator_DNN.pt")
    torch.save(model["discriminator"].state_dict(), "g10_time1_discriminator_DNN.pt")
