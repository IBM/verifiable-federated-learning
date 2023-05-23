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
Script to obtain and partition the data between clients
"""
import os
import json
from typing import List, Tuple, TYPE_CHECKING

import numpy as np
from torchvision import datasets
from fl.client import Client

if TYPE_CHECKING:
    from main import ConfigDict


def setup_femnist_clients(config_dic: "ConfigDict") -> List[Client]:
    """
    Setting up clients for the femnist dataset.

    If the dataset is set to merged_femnist certain classes corresponding to upper-case and lower-case letters
    are combined.

    :params config_dic: Dictionary defining the configuration of the experiment
    :return: list of clients with the femnist/merged_femnist data loaded.
    """
    train_path = "../leaf/data/femnist/data/train"
    test_path = "../leaf/data/femnist/data/test"

    list_of_train_data_files = sorted(
        [f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))]
    )
    list_of_test_data_files = sorted([f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))])

    print(list_of_train_data_files)
    print(list_of_test_data_files)

    list_of_clients = []
    for train_json, test_json in zip(list_of_train_data_files, list_of_test_data_files):
        with open(os.path.join(train_path, train_json)) as file_open:
            train_data_json = json.load(file_open)

        with open(os.path.join(test_path, test_json)) as file_open:
            test_data_json = json.load(file_open)

        for user in train_data_json["users"]:
            user_traindata = train_data_json["user_data"][user]
            user_testdata = test_data_json["user_data"][user]
            list_of_clients.append(
                Client(user, train_data=user_traindata, eval_data=user_testdata, config_dic=config_dic)
            )
            print(
                f"Setup clients {len(list_of_clients)}/3500 with {len(user_traindata['y'])} datapoints",
                flush=True,
            )

    return list_of_clients


def setup_mnist_clients(config_dic: "ConfigDict") -> List[Client]:
    """
    Setting up clients using the mnist data. The data is split evenly between the clients.

    :param config_dic: Dictionary defining the configuration of the experiment
    :return: list of clients with the mnist data loaded.
    """
    (x_train, y_train), (x_test, y_test) = get_minst_data()
    list_of_clients = []

    train_data_per_client = int(len(x_train) / config_dic["num_clients"])
    test_data_per_client = int(len(x_test) / config_dic["num_clients"])

    for i in range(config_dic["num_clients"]):
        train_data = {
            "x": x_train[i * train_data_per_client : (i + 1) * train_data_per_client],
            "y": y_train[i * train_data_per_client : (i + 1) * train_data_per_client],
        }

        eval_data = {
            "x": x_test[i * test_data_per_client : (i + 1) * test_data_per_client],
            "y": y_test[i * test_data_per_client : (i + 1) * test_data_per_client],
        }

        list_of_clients.append(Client(client_id=i, train_data=train_data, eval_data=eval_data, config_dic=config_dic))

    return list_of_clients


def get_minst_data(model_type="cnn") -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Get the MNIST data.

    :param model_type: either cnn or dense. If the model is a cnn the channel dimension is inserted into the data.
                       If the model is a dense one then the data is flattened.
    :return: minst train/test data.
    """
    train_set = datasets.MNIST("./data", train=True, download=True)
    test_set = datasets.MNIST("./data", train=False, download=True)

    x_train = train_set.data.numpy().astype(np.float32)
    y_train = train_set.targets.numpy()

    x_test = test_set.data.numpy().astype(np.float32)
    y_test = test_set.targets.numpy()

    if model_type == "dense":
        x_train = np.reshape(x_train, (x_train.shape[0], -1))
        x_test = np.reshape(x_test, (x_test.shape[0], -1))
    elif model_type == "cnn":
        x_train = np.expand_dims(x_train, axis=1)
        x_test = np.expand_dims(x_test, axis=1)
    else:
        raise ValueError("model_type must be either dense or cnn")

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return (x_train, y_train), (x_test, y_test)


def get_cifar_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Get CIFAR-10 data.

    :return: cifar train/test data.
    """
    train_set = datasets.CIFAR10("./data", train=True, download=True)
    test_set = datasets.CIFAR10("./data", train=False, download=True)

    x_train = train_set.data.astype(np.float32)
    y_train = np.asarray(train_set.targets)

    x_test = test_set.data.astype(np.float32)
    y_test = np.asarray(test_set.targets)

    x_train = np.moveaxis(x_train, [3], [1])
    x_test = np.moveaxis(x_test, [3], [1])

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return (x_train, y_train), (x_test, y_test)


def setup_cifar_clients(config_dic: "ConfigDict") -> List[Client]:
    """
    Setting up clients using the cifar10 data. The data is split evenly between the clients.

    :param config_dic: Dictionary defining the configuration of the experiment
    :return: list of clients with the mnist data loaded.
    """
    (x_train, y_train), (x_test, y_test) = get_cifar_data()
    list_of_clients = []

    train_data_per_client = int(len(x_train) / config_dic["num_clients"])
    test_data_per_client = int(len(x_test) / config_dic["num_clients"])

    for i in range(config_dic["num_clients"]):
        train_data = {
            "x": x_train[i * train_data_per_client : (i + 1) * train_data_per_client],
            "y": y_train[i * train_data_per_client : (i + 1) * train_data_per_client],
        }

        eval_data = {
            "x": x_test[i * test_data_per_client : (i + 1) * test_data_per_client],
            "y": y_test[i * test_data_per_client : (i + 1) * test_data_per_client],
        }

        list_of_clients.append(Client(client_id=i, train_data=train_data, eval_data=eval_data, config_dic=config_dic))

    return list_of_clients
