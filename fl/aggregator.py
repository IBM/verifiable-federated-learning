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
Here is the implementation of the aggregator in the Verification Via Commitments protocol as first proposed in
Verifiable Federated Learning: https://openreview.net/pdf?id=0HIa3HIyIHN
"""
import datetime
import time
import torch

from typing import List, Tuple, TYPE_CHECKING


if TYPE_CHECKING:
    from type_utils import ConfigDict, Commitment


class Aggregator:
    """
    Class providing the implementation of the aggregator in the Verification Via Commitments protocol first proposed in
    Verifiable Federated Learning: https://openreview.net/pdf?id=0HIa3HIyIHN
    """

    def __init__(self):
        pass

    @classmethod
    def fed_sum(
        cls, global_model: torch.nn.Module, client_models: List[dict], config_dic: "ConfigDict"
    ) -> torch.nn.Module:
        """
        Aggregator performing insecure sum without any verification.

        :param global_model: The global pytorch model architecture.
        :param client_models: List of pytorch state_dics with the client model parameters after training.
        :param config_dic: Configuration dictionary.
        :return: The new global model with the summed weights of the participating clients without any verification.
        """
        for param_tensor in global_model.state_dict():
            new_weights = [state_dict[param_tensor].data for state_dict in client_models]
            new_tensor_weights = torch.sum(torch.stack(new_weights), dim=0)

            if config_dic["check_rounding"]:
                new_tensor_weights = torch.round(new_tensor_weights, decimals=config_dic["encoder_base"])

            global_model.state_dict()[param_tensor][:] = new_tensor_weights.detach().clone()

        return global_model

    @classmethod
    def secure_fed_sum(
        cls,
        malicious_aggregator: bool,
        global_model: torch.nn.Module,
        client_models: List[dict],
        com_model_clients: List[List["Commitment"]],
        config_dic: "ConfigDict",
    ) -> Tuple[torch.nn.Module, List["Commitment"], float]:
        """
        Aggregator performs a sum with verification

        :param malicious_aggregator: True or False if to simulate the aggregator tampering with the weights.
        :param global_model: pytorch model with the current global model from the start of the round.
        :param client_models: A list holding pytorch state_dic for the clients that participated in this FL round.
        :param com_model_clients: A nested list, containing the weight commitments per client
        :param config_dic: A dictionary containing the configuration for the experiment.
        :return: A Tuple containing:
                 1) The global model with summed weights.
                 2) A list of Commitments corresponding to each weight.
                 3) The total time taken for the computations.
        """
        if not config_dic["secure"]:
            raise ValueError("secure_fed_sum accessed without the configuration specifying secure")

        time_pre = datetime.datetime.now()
        for param_tensor in global_model.state_dict():
            new_weights = [state_dict[param_tensor].data for state_dict in client_models]
            new_tensor_weights = torch.stack(new_weights)
            if malicious_aggregator:
                new_tensor_weights = torch.sum(new_tensor_weights, dim=0)
                # example attack: the aggregator tampers with one of the weights.
                if param_tensor == "fc1.weight":
                    new_tensor_weights = new_tensor_weights[0, 0] + 0.1
            else:
                new_tensor_weights = torch.round(
                    torch.sum(new_tensor_weights, dim=0), decimals=config_dic["encoder_base"]
                )

            global_model.state_dict()[param_tensor][:] = new_tensor_weights.detach().clone()

        time_aggregator = (datetime.datetime.now() - time_pre).total_seconds()
        com_aggregator_model, time_aggregator_com = cls.compute_commitment(com_model_clients)

        return global_model, com_aggregator_model, time_aggregator + time_aggregator_com

    @staticmethod
    def compute_commitment(com_model_clients: List[List["Commitment"]]) -> Tuple[List["Commitment"], float]:
        """
        Aggregator computes their commitment via multiplication of the supplied client commitments.

        :param com_model_clients: A nested list, containing the weight commitments per client
        :return: A tuple containing the commitments per weight and the time taken for the computations
        """
        n_client = len(com_model_clients)
        num_weights = len(com_model_clients[0])
        com_aggregator_model = []
        start_time = time.time()
        for i in range(num_weights):
            for c_num in range(n_client):
                if c_num == 0:
                    mul = com_model_clients[c_num][i]
                else:
                    mul *= com_model_clients[c_num][i]
            com_aggregator_model.append(mul)
        return com_aggregator_model, time.time() - start_time
