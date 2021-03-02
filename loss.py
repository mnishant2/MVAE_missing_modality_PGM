import os
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from itertools import combinations
from tqdm import tqdm
def elbo_loss(recon_image1=None, image1=None,recon_image2=None, image2=None, recon_speech=None, speech=None,recon_label=None,y=None, mu=None, logvar=None,
              lambda_image=1.0, lambda_speech=1.0, annealing_factor=1):
    """Compute the ELBO for an arbitrary number of data modalities.
    @param recon: list of torch.Tensors/Variables
                  Contains one for each modality.
    @param data: list of torch.Tensors/Variables
                 Size much agree with recon.
    @param mu: Torch.Tensor
               Mean of the variational distribution.
    @param logvar: Torch.Tensor
                   Log variance for variational distribution.
    @param lambda_image: float [default: 1.0]
                         weight for image BCE
    @param lambda_attr: float [default: 1.0]
                        weight for attribute BCE
    @param annealing_factor: float [default: 1]
                             Beta - how much to weight the KL regularizer.
    """
    # assert len(recon) == len(data), "must supply ground truth for every modality."
    # n_modalities = len(recon)
    batch_size   = mu.size(0)


    image1_bce,image2_bce,speech_mse,label_ce  = 0,0,0,0  # reconstruction cost
    if recon_image1 is not None and image1 is not None:
        # image1_bce = torch.mean(torch.sum(binary_cross_entropy_with_logits(
        #     recon_image1.view(-1, 1 * 28 * 28), 
        #     image1.view(-1, 1 * 28 * 28)), dim=1))
        image1_bce= torch.mean(torch.sum(nn.BCEWithLogitsLoss(reduction='none')(
            recon_image1.view(-1, 1 * 28 * 28), 
            image1.view(-1, 1 * 28 * 28)),dim=1),dim=0)
        # image1_bce= torch.mean(torch.sum(nn.MSELoss(reduction='none')(
        #     recon_image1.view(-1, 1 * 28 * 28), 
        #     image1.view(-1, 1 * 28 * 28)),dim=1),dim=0)
    # print("IMAGE1 RECONSTRUCTION",image1_bce)
    if recon_image2 is not None and image2 is not None:
        # image2_bce = torch.mean(torch.sum(binary_cross_entropy_with_logits(
        #     recon_image2.view(-1, 1 * 28 * 28), 
        #     image2.view(-1, 1 * 28 * 28)), dim=1))
        image2_bce= torch.mean(torch.sum(nn.BCEWithLogitsLoss(reduction='none')(
            recon_image2.view(-1, 1 * 28 * 28), 
            image2.view(-1, 1 * 28 * 28)),dim=1),dim=0)
        # image2_bce= torch.mean(torch.sum(nn.MSELoss(reduction='none')(
        #     recon_image2.view(-1, 1 * 28 * 28), 
        #     image2.view(-1, 1 * 28 * 28)),dim=1),dim=0)
    # print("IMAGE2 RECONSTRUCTION",image2_bce)    
    if recon_speech is not None and speech is not None:  # this is for an attribute
        # print(recon_speech,speech, type(recon_speech),type(speech))
        loss= nn.MSELoss(reduction='none')
        speech_mse = torch.mean(torch.sum(loss(recon_speech, speech),dim=1),dim=0)
    # print("SPEECH RECONSTRUCTION",speech_mse)
    if recon_label is not None and y is not None:
        label_ce=torch.mean(nn.CrossEntropyLoss(reduction='none')(recon_label,torch.argmax(y,dim=1)),dim=0)
    # print("CLASSIFICATION",label_ce)
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1),dim=0)
    # print("KL DIVERGENCE",KLD)
    # KLD=0
    ELBO = lambda_image * image1_bce + lambda_image * image2_bce + lambda_speech * speech_mse+label_ce + annealing_factor * KLD
    return ELBO,image1_bce,image2_bce,speech_mse,label_ce