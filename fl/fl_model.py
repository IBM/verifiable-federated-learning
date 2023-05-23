# Copyright (C) 2022 Verifiable Federated Learning Authors
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#  http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Script with the different models we use for the various datasets
"""
import torch
from torch import nn
import numpy as np


class MNISTModel(nn.Module):
    """
    Pytorch model for MNIST training. The smallest neural network we use.
    """

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, num_classes: int):
        super(MNISTModel, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1), dilation=(1, 1), padding=(0, 0)
        )
        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), dilation=(1, 1), padding=(0, 0)
        )

        self.fc1 = nn.Linear(in_features=1024, out_features=512)

        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.DEVICE)
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class FLModel(nn.Module):
    """
    Pytorch model for FEMNIST and merged FEMNIST training. The largest neural network we use.
    """

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, num_classes: int):
        super(FLModel, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1), dilation=(1, 1), padding=(0, 0)
        )
        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), dilation=(1, 1), padding=(0, 0)
        )

        self.fc1 = nn.Linear(in_features=1024, out_features=2048)

        self.fc2 = nn.Linear(in_features=2048, out_features=num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.DEVICE)
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class CIFARModel(nn.Module):
    """
    Pytorch model for CIFAR training.
    """

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, num_classes: int):
        super(CIFARModel, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding=(0, 0)
        )

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding=(0, 0)
        )

        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding=(0, 0)
        )

        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding=(0, 0)
        )

        self.fc1 = nn.Linear(in_features=1600, out_features=1024)

        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.DEVICE)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.max_pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def setup_model(dataset: str) -> "torch.nn.Module":
    """
    Fetches the relevant pytorch model for the dataset.

    :param dataset: Name of the dataset we are training on.
    :return: the model used for federated learning.
    """
    if dataset in {"mnist", "cifar"}:
        num_classes = 10
    elif dataset == "merged_femnist":
        num_classes = 47
    elif dataset == "femnist":
        num_classes = 62
    else:
        raise ValueError("Dataset must be `mnist`, `cifar`, merged_femnist, or `femnist`")

    if dataset == "mnist":
        return MNISTModel(num_classes=num_classes).to(MNISTModel.DEVICE)
    elif dataset in {"merged_femnist", "femnist"}:
        return FLModel(num_classes=num_classes).to(FLModel.DEVICE)
    elif dataset == "cifar":
        return CIFARModel(num_classes=num_classes).to(CIFARModel.DEVICE)
    else:
        raise ValueError("Model not found")
