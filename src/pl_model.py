from math import log10
import numpy as np

import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from src.utils import *

from src.Loss import GANLoss
from src.Discriminator import NLayerDiscriminator
from src.Unet import UnetGenerator
from src.ResNetGenerator import ResnetGenerator
from src.dataset import DatasetFromFolder


class Pix2PixModel(pl.LightningModule):
    def __init__(
        self,
        image_folder,
        input_nc=3,
        output_nc=3,
        ndf=64,
        lr=0.002,
        beta1=0.5,
        weight_decay=0.0005,
        lamb=10,
        ngf=64,
    ):

        # super(Pix2PixModel, self).__init__()
        super().__init__()
        self.save_hyperparameters()
        self.model = UnetGenerator(input_nc=input_nc, output_nc=output_nc, num_downs=5)

        self.train_dataset = DatasetFromFolder(image_folder=image_folder)
        self.netG = self.model

        self.netD = NLayerDiscriminator(input_nc + output_nc, ndf)

        self.criterionGAN = GANLoss()
        self.criterionL1 = nn.L1Loss()
        self.criterionMSE = nn.MSELoss()
        self.lamb = lamb
        self.lr = lr
        self.beta1 = beta1
        self.weight_decay = weight_decay
        self.val_pnsr = 0
        self.checkpoint_path = checkpoint_path

    def forward(self, input):  # メインのネットワークだけに通す
        return self.netG(input)

    def training_step(self, batch, batch_idx, optimizer_idx):
        color_adjusted_image, original_image = batch[0], batch[1]
        image_npy = tensor2image(original_image[0].cpu())
        mask_red = extract_red_area(image_npy, blur_size=(15, 15), contrust_degree=4)
        mask_purple = extract_purple_area(image_npy, blur_size=(5, 5))
        purple_ratio = len(mask_purple[mask_purple == 1]) / len(
            mask_purple[mask_red == 1]
        )

        fake = self.forward(color_adjusted_image)
        fake_ab = torch.cat((color_adjusted_image, fake), 1)
        pred_fake = self.netD.forward(fake_ab)

        # train generator
        if optimizer_idx == 0:
            loss_g_gan = self.criterionGAN(pred_fake, True)
            # Second, G(A) = B
            loss_g_l1_red = self.criterionL1(
                fake[:, :, mask_red == 1], original_image[:, :, mask_red == 1]
            )
            loss_g_l1_purple = (
                self.criterionL1(
                    fake[:, :, mask_purple == 1], original_image[:, :, mask_purple == 1]
                )
                * purple_ratio
            )
            loss_g_l1 = self.criterionL1(fake, original_image)
            loss_g = (
                loss_g_gan
                + (loss_g_l1_red + loss_g_l1_purple) * self.lamb
                + loss_g_l1 * self.lamb / 10
            )
            # print(loss_g)

            self.log(
                "generator loss",
                loss_g_gan,
                prog_bar=True,
                logger=True,
                on_epoch=True,
                on_step=True,
            )

            self.log(
                "L1 loss_red",
                loss_g_l1_red,
                prog_bar=False,
                logger=True,
                on_epoch=True,
                on_step=True,
            )

            self.log(
                "L1 loss",
                loss_g_l1,
                prog_bar=False,
                logger=True,
                on_epoch=True,
                on_step=True,
            )

            return {"loss": loss_g}

        # train discriminator
        if optimizer_idx == 1:

            if np.max(mask_red) == 0:
                mask_red = np.ones_like(mask_red)

            ############################
            # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
            # train with fake
            loss_d_fake = self.criterionGAN(pred_fake, False)

            # train with real
            real_image = torch.cat((color_adjusted_image, original_image), 1)
            pred_real = self.netD.forward(real_image)
            loss_d_real = self.criterionGAN(pred_real, True)
            # Combined loss
            loss_d = (loss_d_fake + loss_d_real) * 0.5

            self.log(
                "Discriminator loss",
                loss_d,
                prog_bar=True,
                logger=True,
                on_epoch=True,
                on_step=True,
            )

            return {"loss": loss_d}

    def validation_step(self, batch, batch_idx):
        color_adjusted_image, original_image = batch[0], batch[1]
        image_npy = tensor2image(original_image[0].cpu())
        mask_red = extract_red_area(image_npy, blur_size=(15, 15), contrust_degree=4)

        if np.max(mask_red) == 0:
            mask_red = np.ones_like(mask_red)
        pred = self.forward(color_adjusted_image)
        mse_red = self.criterionMSE(
            pred[:, :, mask_red == 1], original_image[:, :, mask_red == 1]
        )
        psnr_red = 10 * log10(1 / (mse_red.data.item() + 0.000001))

        mask_purple = extract_purple_area(image_npy, blur_size=(5, 5))
        if np.max(mask_purple) == 0:
            psnr_purple = 10
        else:
            mse_purple = self.criterionMSE(
                pred[:, :, mask_purple == 1], original_image[:, :, mask_purple == 1]
            )
            psnr_purple = 10 * log10(1 / (mse_purple.data.item() + 0.000001))

        purple_ratio = len(mask_purple[mask_purple == 1]) / len(
            mask_purple[mask_red == 1]
        )

        psnr = psnr_red * psnr_purple * purple_ratio * 3

        self.log(
            "psnr",
            psnr,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )

        return {"psnr": torch.tensor(psnr)}

    def validation_epoch_end(self, outputs):

        val_pnsr = torch.stack([x["psnr"] for x in outputs]).mean()
        if self.val_pnsr < val_pnsr:
            self.val_pnsr = val_pnsr
            confirm_output_folder(self.checkpoint_path)
            torch.save(
                self.state_dict(),
                join(self.checkpoint_path, "last_model_{}.pt".format(val_pnsr)),
            )

    def configure_optimizers(self):

        optimizer_G = optim.Adam(
            self.netG.parameters(),
            lr=self.lr,
            betas=(self.beta1, 0.999),
            weight_decay=self.weight_decay,
        )
        optimizer_D = optim.Adam(
            self.netD.parameters(),
            lr=self.lr,
            betas=(self.beta1, 0.999),
            weight_decay=self.weight_decay,
        )
        # scheduler_G = StepLR(optimizer_G, step_size=20, gamma=0.4)
        # scheduler_D = StepLR(optimizer_D, step_size=20, gamma=0.4)
        scheduler_G = ExponentialLR(optimizer_G, gamma=0.85)
        scheduler_D = ExponentialLR(optimizer_D, gamma=0.85)

        return [optimizer_G, optimizer_D], [scheduler_G, scheduler_D]
        # return [optimizer_G, optimizer_D]

    def train_dataloader(self):
        # REQUIRED
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=1,
            num_workers=4,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=1,
            num_workers=4,
            shuffle=True,
        )

    #
    #
    # def training_step(...)
    #
    #
    # def validation_step(...)
    #
    # def validation_step_end(...)
    #
    # def configure_optimizers(self):
    #     # REQUIRED
    #     optimizer = torch.optim.Adam(
    #         self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    #     )
    #     scheduler = lr_scheduler.ExponentialLR(
    #         optimizer, gamma=config.learning_rate_decay
    #     )
    #     return [optimizer], [scheduler]
    #

    #
    # def val_dataloader(self):
    #     return self.data_loader_valid
