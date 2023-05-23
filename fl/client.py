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
Script with the FL client class to handle local training, verification, and commit computation
"""
import numpy as np
import time

import torch

import torchvision
import torchvision.transforms as transforms

from sklearn.utils import shuffle
from typing import Tuple, Dict, List, Union, Optional, TYPE_CHECKING

from commitment.shamir_secret_sharing import ShamirSecretSharing
from commitment.commitment import Commitment

if TYPE_CHECKING:
    from type_utils import ConfigDict, Params


class Client:
    """
    Class which holds a clients local data and training/evaluation methods.
    It does NOT hold the model locally. This avoids needing to store a model per client in GPU memory.
    """

    def __init__(
        self,
        client_id: int,
        train_data: Dict[str, np.ndarray],
        eval_data: Dict[str, np.ndarray],
        config_dic: "ConfigDict",
    ) -> None:
        """
        Performs initial client setup

        :param client_id: Numerical identifier of the client
        :param train_data: Training data held locally by the client
        :param eval_data: Client's evaluation data
        :param config_dic: Dictionary with the experimental configuration
        """
        self.client_id = client_id
        self.bsize = config_dic["bsize"]
        self.augmenter: Optional[torchvision.transforms.transforms.Compose] = None
        self.verbose = False

        train_data_x = np.asarray(train_data["x"]).astype(np.float32)
        if config_dic["dataset"] == "cifar":
            train_data_x = np.reshape(train_data_x, (-1, 3, 32, 32))
            self.augmenter = transforms.Compose(
                [torchvision.transforms.RandomAffine(10, translate=(0.1, 0.1)), transforms.RandomHorizontalFlip(0.5)]
            )
        else:
            train_data_x = np.reshape(train_data_x, (-1, 1, 28, 28))

        y_train = np.asarray(train_data["y"])
        train_data_x, y_train = shuffle(train_data_x, y_train)
        num_of_samples = len(y_train)

        if config_dic["dataset"] == "merged_femnist":
            print("merging")
            y_train = class_merging(y_train)

        self.train_data = (train_data_x[0 : int(num_of_samples * 0.8)], y_train[0 : int(num_of_samples * 0.8)])

        y_test = np.asarray(eval_data["y"])

        if config_dic["dataset"] == "merged_femnist":
            print("merging")
            y_test = class_merging(y_test)

        test_data_x = np.asarray(eval_data["x"]).astype(np.float32)

        if config_dic["dataset"] == "cifar":
            test_data_x = np.reshape(test_data_x, (-1, 3, 32, 32))
        else:
            test_data_x = np.reshape(test_data_x, (-1, 1, 28, 28))

        self.eval_data = (test_data_x, y_test)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config_dic = config_dic

    def train_loop(
        self, model: torch.nn.Module, opt: torch.optim.Optimizer
    ) -> Tuple[torch.nn.Module, torch.optim.Optimizer, List[np.ndarray], List[np.ndarray]]:
        """
        Trains the model over the client's data

        :param model: The model sent by the aggregator.
        :param opt: Optimiser sent by the aggregator. If this is a stateful optimizer it may have parameters as well.
        :return: A Tuple containing
                1) The trained model
                2) The optimizer
                3) A list containing the per batch accuracies
                4) A list containing the per batch losses
        """
        (x_train, y_train) = self.train_data
        model.train()
        num_of_batches = int(len(x_train) / self.bsize)

        acc = []
        loss_list = []
        x_train, y_train = shuffle(x_train, y_train)

        for bnum in range(num_of_batches):
            x_batch = torch.from_numpy(np.copy(x_train[self.bsize * bnum : (bnum + 1) * self.bsize])).to(self.device)
            y_batch = (
                torch.from_numpy(np.copy(y_train[self.bsize * bnum : (bnum + 1) * self.bsize]))
                .type(torch.LongTensor)
                .to(self.device)
            )

            if self.config_dic["dataset"] == "cifar" and self.config_dic["data_augmentation"] and self.augmenter is not None:
                x_batch = self.augmenter(x_batch)

            # zero the parameter gradients
            opt.zero_grad()
            outputs = model(x_batch)

            loss = self.criterion(outputs, y_batch)
            loss.backward()
            opt.step()

            preds = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            y_batch = y_batch.cpu().numpy()
            acc.append(np.mean(preds == y_batch))

            loss_list.append(loss.item())
            if bnum > 0 and bnum % 500 == 0 and self.verbose:
                print(f"Bnum {bnum}: Loss {loss.item()} Acc {np.mean(acc)}")
        if self.verbose:
            print(f"Final client performance: Bnum {bnum}: Loss {loss.item()} Acc {np.mean(acc)}")
        return model, opt, acc, loss_list

    def eval_model(self, model: torch.nn.Module) -> Tuple[np.ndarray, np.ndarray, int]:
        (x_eval, y_eval) = self.eval_data
        model.eval()
        acc_list = []
        loss_list = []

        num_of_batches = int(len(x_eval) / self.bsize)
        if num_of_batches == 0:
            x_batch = torch.from_numpy(np.copy(x_eval[0:])).to(self.device)
            y_batch = torch.from_numpy(np.copy(y_eval[0:])).type(torch.LongTensor).to(self.device)

            outputs = model(x_batch)
            loss = self.criterion(outputs, y_batch)
            outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1)

            acc = np.mean(outputs == y_batch.cpu().detach().numpy()) * 100
            acc_list.append(acc)
            loss_list.append(loss.cpu().detach())
        else:
            for bnum in range(num_of_batches):
                x_batch = torch.from_numpy(np.copy(x_eval[bnum * self.bsize : (bnum + 1) * self.bsize])).to(self.device)
                y_batch = (
                    torch.from_numpy(np.copy(y_eval[bnum * self.bsize : (bnum + 1) * self.bsize]))
                    .type(torch.LongTensor)
                    .to(self.device)
                )

                outputs = model(x_batch)
                loss = self.criterion(outputs, y_batch)
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1)

                acc = np.mean(outputs == y_batch.cpu().detach().numpy()) * 100
                acc_list.append(acc)
                loss_list.append(loss.cpu().detach())

        return np.mean(loss_list), np.mean(acc_list), len(x_eval)

    @staticmethod
    def flatten_model(client_state_dict: dict) -> np.ndarray:
        """
        Helper function to flatten all the model parameters into a single numpy array to make the computations
        of the commitments simpler from a coding perspective.

        :param client_state_dict: pytorch state dict for the model
        :return: a numpy array with the flattened client weights
        """
        flattened_model_weights = []
        for param_tensor in client_state_dict:
            flattened_model_weights.append(client_state_dict[param_tensor].detach().cpu().numpy().flatten())
        return np.concatenate(flattened_model_weights)

    def compute_commitments(
        self, params: "Params", client_state_dict: Union[dict, np.ndarray], r: int
    ) -> Tuple[List[Commitment], float]:
        """
        Computes the commitments of the weights after initial training.

        :param params: Public parameters for the commitments
        :param client_state_dict: torch state dict or a flattened model (numpy array)
        :param r: secret random number r
        :return: A Tuple containing the commitments over the flattened model parameters and the time taken.
        """

        if not isinstance(client_state_dict, np.ndarray):
            flattened_model_weights = self.flatten_model(client_state_dict)
            start_time = time.time()
            com_model = []
            for weight in flattened_model_weights:
                com_model.append(Commitment.commit(params, round(weight, self.config_dic["encoder_base"]), r))
        else:
            start_time = time.time()
            com_model = []
            for weight in client_state_dict:
                com_model.append(Commitment.commit(params, round(weight, self.config_dic["encoder_base"]), r))

        return com_model, time.time() - start_time

    @staticmethod
    def compute_secret(shares: List[Tuple[int, bytes]]) -> int:
        """
        Computes the secret sum of r using the shares and Shamir's Secret Sharing protocol

        :param shares: shares needed to recompute the secret.
        :return: the secret sum of client r
        """
        return ShamirSecretSharing.compute_secret(shares)

    def verify_commitments(
        self, shares: List[Tuple[int, bytes]], aggregator_model: dict, com_aggregator_model: List[Commitment]
    ) -> Tuple[bool, float]:
        """
        Verifies the commitments supplied by the aggregator against the supplied weights.

        :param shares: at least k shares required by Shamir Secret Sharing to recompute the secret.
        :param aggregator_model: The model supplied by the aggregator.
        :param com_aggregator_model: A list containing the weight commitments from the aggregator
        :return:
        """
        flattened_model_weights = self.flatten_model(aggregator_model)

        start_time = time.time()
        secret = self.compute_secret(shares)
        check = True
        for i, t in enumerate(flattened_model_weights):
            verify = com_aggregator_model[i].verify(round(t, self.config_dic["encoder_base"]), secret)
            if not verify:
                check = False
                print("----------------------------------------------------")
                print("Aggregator tampering Detected. Stopping Computation.")
                print("----------------------------------------------------")
                exit()
        return check, time.time() - start_time


def class_merging(labels):
    """
    Classes we merge:
        c class 38 with C class 12
        i class 44 with I class 18
        j class 45 with J class 19
        k class 46 with K class 20
        l class 47 with L class 21
        m class 48 with M class 22
        o class 50 with O class 24
        p class 51 with P class 25
        s class 54 with S class 28
        u class 56 with U class 30
        v class 57 with V class 31
        w class 58 with W class 32
        x class 59 with X class 33
        y class 60 with Y class 34
        z class 61 with Z class 35
    """
    class_labels = [38, 44, 45, 46, 47, 48, 50, 51, 54, 56, 57, 58, 59, 60, 61]

    for i in range(len(labels)):
        # merge class label
        if labels[i] in class_labels:
            labels[i] = labels[i] - 26
        else:
            # adjust labels so that class numbers are continuous between 0 - 46
            # count the number of classes which were removed
            # below the current samples
            num_of_missing_classes = np.sum(np.where(labels[i] > class_labels, 1, 0))
            labels[i] = labels[i] - num_of_missing_classes
    # print(np.amax(labels))
    # assert np.amax(labels) == 46
    return labels
