# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

import timm
import torchvision.transforms as transforms
import copy

from backbone import MammothBackbone, register_backbone


class LEAR(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, num_classes: int) -> None:
        """
        Instantiates the layers of the network.
        """

        super(LEAR, self).__init__()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.num_classes = num_classes

        self.model_dim = 768
        self.fc_dim = 500

        self.fcArr = nn.ModuleList([nn.Linear(self.model_dim, self.fc_dim, device=self.device)])
        self.classifierArr = nn.ModuleList([nn.Linear(self.fc_dim, self.num_classes, device=self.device)])
        self.distributions = []

        model_name_vit = 'vit_base_patch16_224'# 'vit_base_patch16_224.augreg_in21k' # 'vit_base_patch16_224'

        self.c_expert = 0

        self.vitProcess = transforms.Compose(
            [transforms.Resize(224)])

        self.local_vitmodel = timm.create_model(
            model_name_vit,
            pretrained=True,
            num_classes=self.num_classes
        )

        self.feature_extractor = timm.create_model(
            model_name_vit,
            pretrained=True,
            num_classes=self.num_classes
        )

        for param in self.local_vitmodel.parameters():
            param.requires_grad = False

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # partially activated
        # num_unfrozen_layers = 3
        # for block_index, block in enumerate(self.local_vitmodel.blocks):
        #     if block_index in [9, 10, 11]:
        #         for param in block.parameters():
    
    def extract_feature(self, processX):
        features = self.feature_extractor.patch_embed(processX)
        cls_token = self.feature_extractor.cls_token.expand(features.shape[0], -1, -1)
        features = torch.cat((cls_token, features), dim=1)
        features = features + self.feature_extractor.pos_embed

        # forward pass till -3
        for block in self.feature_extractor.blocks:
            features = block(features)
        features = self.feature_extractor.norm(features)
        class_token = features[:, 0, :]
        return class_token
    
    def CreateNewExper(self, idx, num_classes):
        new_fc_dim = self.fc_dim
        new_fc = nn.Linear(self.model_dim, new_fc_dim, device=self.device)
        # self.classifier.load_state_dict(self.classifierArr[idx].state_dict())
        # print('load expert ' + str(idx + 1) + ' parameters')
        self.fcArr.append(new_fc)
        self.classifier = nn.Linear(new_fc_dim, num_classes, device=self.device)
        self.classifierArr.append(self.classifier)

        print('Create new expert ' + str(len(self.classifierArr)))

        self.c_expert = len(self.classifierArr) - 1

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)

    def forward_expert(self, local_features, return_features=False):
        fcfeatures = self.fcArr[self.c_expert](local_features)
        out = self.classifierArr[self.c_expert](fcfeatures)
        return out

    def forward(self, x: torch.Tensor, return_features=False) -> torch.Tensor:
        local_features = self.forward_fusion(x, return_features=True)
        outputs = self.forward_expert(local_features, return_features=return_features)
        return  outputs, local_features

    def forward_fusion(self, x, return_features=False):

        processX = self.vitProcess(x)
        if processX.size(1) == 1:
            processX = processX.expand(-1, 3, -1, -1)
        local_features = self.local_vitmodel.patch_embed(processX)
        local_cls_token = self.local_vitmodel.cls_token.expand(local_features.shape[0], -1, -1)
        local_features = torch.cat((local_cls_token, local_features), dim=1)
        local_features = local_features + self.local_vitmodel.pos_embed

        # forward pass till -3 layer
        for block in self.local_vitmodel.blocks:
            local_features = block(local_features)

        local_features = self.local_vitmodel.norm(local_features)
        local_features = local_features[:, 0, :]
        return local_features

    def myprediction(self, x, index):
        with torch.no_grad():
            local_features = self.forward_fusion(x)
            fcfeatures = self.fcArr[index](local_features)
            out = self.classifierArr[index](fcfeatures)
            return out


@register_backbone("lear")
def LEAR_backbone(num_classes):
    return LEAR(num_classes)





