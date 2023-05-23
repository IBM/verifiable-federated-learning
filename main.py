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
Main file to replicate the experiments using Verification Via Commitments (VVC)
in the Verifiable Federated Learning paper: https://openreview.net/pdf?id=0HIa3HIyIHN
"""
import os
import datetime
import copy
import concurrent.futures
import time
from typing import TypedDict, List, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch import optim

from util import Logger
from fl.setup import Setup
from fl.data import setup_mnist_clients, setup_femnist_clients, setup_cifar_clients
from fl.aggregator import Aggregator
from fl.fl_model import setup_model

if TYPE_CHECKING:
    from fl.client import Client
    from type_utils import Params


class ResumeInfo(TypedDict):
    """
    A TypedDict class to define the types in the resume form checkpoint dictionary.
    """

    to_resume: bool
    resume_run: Optional[str]
    resume_round: Optional[int]


class ConfigDict(TypedDict):
    """
    A TypedDict class to define the types in the configuration dictionary.
    """

    dataset: str
    num_clients: int
    clients_participating_per_round: int
    fl_rounds: int
    learning_rate: float
    bsize: int
    malicious_clients: int
    secure: bool
    malicious_aggregator: bool
    num_colluding_clients: int
    data_augmentation: bool
    check_rounding: bool
    encoder_base: int
    to_log: bool
    resume_info: ResumeInfo
    save_path: str
    key_size: int


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_clients(config_dic: ConfigDict) -> List["Client"]:
    """
    Gets a list of fl clients as specified in the configuration dictionary

    :param config_dic: configuration dictionary
    :return: List of clients used for the experiment
    """
    if config_dic["dataset"] in ["femnist", "merged_femnist"]:
        return setup_femnist_clients(config_dic)
    if config_dic["dataset"] == "mnist":
        return setup_mnist_clients(config_dic)
    if config_dic["dataset"] == "cifar":
        return setup_cifar_clients(config_dic)
    raise ValueError("Provided Dataset does not match")


def round_tensor_values(state_dict: dict, config_dic: ConfigDict) -> dict:
    """
    Round tensors to a number of decimal places governed by config_dic['encoder_base']

    :param state_dict: pytorch model's state dict
    :param config_dic: configuration dictionary
    :return: pytorch state dict with rounded tensor values
    """
    rounded_dict = {}
    for param in state_dict:
        rounded_dict[param] = torch.round(state_dict[param], decimals=config_dic["encoder_base"])
    return rounded_dict


def average_summed_values(model: torch.nn.Module, clients_participating_per_round: int) -> torch.nn.Module:
    """
    As the protocol supports only secure sums, the division is moved here to be performed post-aggregation locally
    by the clients.
    :param model: the global model with weights which were summed in the aggregation.
    :param clients_participating_per_round: number of clients which participated in the fl training round, not the total
    number of clients
    :return: the global model with weights which are now averaged
    """
    divided_vals = {}
    state_dict = model.state_dict()
    for param in model.state_dict():
        divided_vals[param] = state_dict[param] / clients_participating_per_round
    model.load_state_dict(divided_vals)
    return model


def compute_parallel_commits(
    list_of_clients: List["Client"],
    params: "Params",
    flattened_clients: List[np.ndarray],
    r_i_rounds: List[List[int]],
    round_num: int,
    config_dic: ConfigDict,
) -> List[concurrent.futures._base.Future]:
    """
    Parallelize the commitments over the number of clients.

    :param list_of_clients: List containing all the clients. We just need the compute_commitments method from any client
    :param params: A Params object containing public parameters g, h, and q
    :param flattened_clients: List of flattened client with weights as numpy array
    :param r_i_rounds: Random numbers per round per client
    :param round_num: The current FL round number
    :param config_dic: Dictionary defining the configuration of the experiment
    :return: commitment list from ProcessPoolExecutor
    """
    max_workers = config_dic["clients_participating_per_round"]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = [
            executor.submit(
                list_of_clients[i].compute_commitments, params, flattened_clients[i], r_i_rounds[round_num][i]
            )
            for i in range(max_workers)
        ]
    concurrent.futures.wait(results)
    return results


def fl_loop(
    config_dic: ConfigDict,
    params: Optional["Params"],
    shamir_k: Optional[int],
    r_i_rounds: Optional[List[List[int]]],
    shares_rounds: Optional[List[List[Tuple[int, bytes]]]],
    list_of_clients: List["Client"],
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    logger: Optional[Logger] = None,
) -> None:
    """
    The primary function which runs the experiments

    :param config_dic: the configuration that parametrises the experiments
    :param params: A Params object containing public parameters g, h, and q
    :param shamir_k: The k value for the shamir secret sharing protocol.
                     Needs to be greater than the number of malicious clients.
    :param r_i_rounds: Random numbers per round per client
    :param shares_rounds: The pre-computed shares, per round of the secret.
    :param list_of_clients: List containing all the clients
    :param model: pytorch model architecture
    :param opt: pytorch optimizer
    :param logger: logging utility object to save models and results.
    :return: None
    """
    client_times: Dict[str, dict] = {}
    client_verification_times: Dict[str, Dict[str, float]] = {}
    aggregator_times = {}

    check_frequency = 100
    if config_dic["dataset"] == "mnist":
        check_frequency = 1

    init_fl_round = 0
    if config_dic["resume_info"]["to_resume"] and config_dic["resume_info"]["resume_round"] is not None:
        init_fl_round = config_dic["resume_info"]["resume_round"] + 1

    for round_num in range(init_fl_round, config_dic["fl_rounds"]):
        start_time_of_round = time.time()
        client_times[str(round_num)] = {}
        client_verification_times[str(round_num)] = {}

        initial_state_dic = copy.deepcopy(model.state_dict())
        initial_opt_state_dic = copy.deepcopy(opt.state_dict())

        client_models = []
        client_opts = []
        client_com = []

        round_acc = []
        round_loss = []

        client_participating_index = np.random.choice(
            a=len(list_of_clients), size=config_dic["clients_participating_per_round"], replace=False
        )
        flattened_clients = []
        for cnum in client_participating_index:
            model.load_state_dict(copy.deepcopy(initial_state_dic))
            opt.load_state_dict(copy.deepcopy(initial_opt_state_dic))
            model, opt, client_acc, client_loss = list_of_clients[cnum].train_loop(model=model, opt=opt)

            client_models.append(copy.deepcopy(model.state_dict()))
            client_opts.append(copy.deepcopy(opt.state_dict()))

            # compute commitment
            if config_dic["secure"] or config_dic["check_rounding"]:
                client_models[-1] = round_tensor_values(client_models[-1], config_dic)
            if config_dic["secure"]:
                flattened_clients.append(
                    list_of_clients[cnum].flatten_model(client_models[-1])
                )  # due to cuda we need to do this here

            round_acc.append(client_acc)
            round_loss.append(client_loss)
        if config_dic["secure"] and params is not None and r_i_rounds is not None:
            print("Starting client commits", flush=True)
            start_time_of_commits = time.time()
            results = compute_parallel_commits(
                list_of_clients, params, flattened_clients, r_i_rounds, round_num, config_dic
            )
            for res in results:
                for check in res.result():
                    if isinstance(check, list):
                        client_com.append(check)

            print("Total time when pooling ", time.time() - start_time_of_commits)
            for cnum in client_participating_index:
                client_times[str(round_num)][str(cnum)] = time.time() - start_time_of_commits

            if logger is not None:
                logger.log_times(client_times, file_name="client_times.json")

        round_acc = np.concatenate(round_acc)
        round_loss = np.concatenate(round_loss)

        print(f"End of round {round_num}: loss {np.mean(round_loss)} acc {np.mean(round_acc)*100}", flush=True)
        end_time_of_round = time.time()
        if logger is not None:
            logger.log_results(
                list(map(str, [round_num, end_time_of_round - start_time_of_round])), file_name="round_times.csv"
            )

        if config_dic["secure"]:
            print("Aggregator performing Secure Agg")
            model, com_aggregator_model, aggregator_time = Aggregator.secure_fed_sum(
                config_dic["malicious_aggregator"], model, client_models, client_com, config_dic
            )
            aggregator_times[str(round_num)] = aggregator_time
        else:
            time_pre = datetime.datetime.now()
            model = Aggregator.fed_sum(model, client_models, config_dic)
            aggregator_times[str(round_num)] = (datetime.datetime.now() - time_pre).total_seconds()

        if logger is not None:
            logger.log_times(aggregator_times, file_name="aggregator_times.json")

        if (
            config_dic["secure"]
            and shares_rounds is not None
            and (round_num % check_frequency == 0 or config_dic["malicious_aggregator"])
        ):
            time_pre = datetime.datetime.now()
            shares = shares_rounds[round_num][0:shamir_k]

            client_shares_time = (datetime.datetime.now() - time_pre).total_seconds()

            for cnum in client_participating_index:
                print(f"Client {cnum} checking commits ", flush=True)
                check, client_verification_time_2 = list_of_clients[cnum].verify_commitments(
                    shares, model.state_dict(), com_aggregator_model
                )
                client_verification_times[str(round_num)][str(cnum)] = client_verification_time_2 + client_shares_time
                if not check:
                    print("Aggregator cheat!")
                if logger is not None:
                    logger.log_times(client_verification_times, file_name="client_verification_times.json")

        model = average_summed_values(
            model=model, clients_participating_per_round=config_dic["clients_participating_per_round"]
        )

        if (round_num % 5 == 0 and round_num > 0) or round_num == config_dic["fl_rounds"] - 1:
            running_test_loss, running_test_acc = compute_test_statistics(list_of_clients, model)
            print(f"On round {round_num} test loss {running_test_loss}, test acc {running_test_acc}", flush=True)
            if logger is not None:
                logger.log_results(list(map(str, [round_num, running_test_loss, running_test_acc])))
                logger.log_results(
                    list(map(str, [round_num, np.mean(round_loss), np.mean(round_acc)])), file_name="train_results.csv"
                )

                logger.save_models(model=model, opt=opt)


def compute_test_statistics(list_of_clients: List["Client"], model: torch.nn.Module) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the average test loss and accuracy weighted by the number of samples each client holds

    :param list_of_clients: list of all clients
    :param model: global model to evaluate
    :return: average test loss and average test accuracy
    """
    running_test_loss = []
    running_test_acc = []
    weighting = []
    for client in list_of_clients:
        test_loss, test_acc, num_samples = client.eval_model(model=model)
        running_test_loss.append(test_loss)
        running_test_acc.append(test_acc)
        weighting.append(num_samples)
    return np.average(running_test_loss, weights=weighting), np.average(running_test_acc, weights=weighting)


def init(
    config_dic: ConfigDict, logger: Optional[Logger] = None
) -> Tuple[
    Optional["Params"],
    Optional[int],
    Optional[List[List[int]]],
    Optional[List[List[Tuple[int, bytes]]]],
    List["Client"],
    torch.nn.Module,
    torch.optim.Optimizer,
]:
    """
    Performs initial setup generating 1) the clients, 2) the model, and 3) parameters required by the protocol

    :param config_dic: Dictionary defining the configuration of the experiment
    :param logger: A utility class to save data and provide checkpointing info.
    :return: The parameters required for the protocol, as well as the clients with their data, and the torch model and
             optimizer.
    """
    time_pre = datetime.datetime.now()
    list_of_clients = get_clients(config_dic)
    model = setup_model(dataset=config_dic["dataset"])

    if config_dic["resume_info"]["to_resume"] and logger is not None:
        model_path = logger.fetch_model_resume_file(config_dic)
        print("loading model from ", model_path)
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        print("loading model weights")
        model.load_state_dict(checkpoint["model_head"])

    print("Total model parameters ", sum(p.numel() for p in model.parameters()))
    opt = optim.SGD(model.parameters(), lr=config_dic["learning_rate"])
    if config_dic["resume_info"]["to_resume"]:
        opt.load_state_dict(checkpoint["opt"])

    # Trusted Setup
    if config_dic["secure"]:
        shamir_n = config_dic["clients_participating_per_round"]
        shamir_k = config_dic["malicious_clients"] + 1

        params, r_i_rounds, shares_rounds = Setup.setup(
            config_dic["key_size"], shamir_n, shamir_k, config_dic["fl_rounds"]
        )

        if logger is not None:
            logger.save_time("secure", time_pre)
        return params, shamir_k, r_i_rounds, shares_rounds, list_of_clients, model, opt
    if logger is not None:
        logger.save_time("vanilla", time_pre)
    return None, None, None, None, list_of_clients, model, opt


def main(config_dic: ConfigDict) -> None:
    """
    Main that runs the experiments parameterised by the configuration dictionary.

    :param config_dic: Dictionary defining the configuration of the experiment
    :return: None
    """
    if config_dic["to_log"]:
        logger = Logger(config_dic)
    else:
        logger = None

    params, shamir_k, r_i_rounds, shares_rounds, list_of_clients, model, opt = init(config_dic, logger)
    fl_loop(config_dic, params, shamir_k, r_i_rounds, shares_rounds, list_of_clients, model, opt, logger)


if __name__ == "__main__":
    resume_info: ResumeInfo = {"to_resume": False, "resume_run": None, "resume_round": None}

    configuration: ConfigDict = {
        "dataset": "mnist",
        "num_clients": 20,  # not applicable for femnist
        "clients_participating_per_round": 10,
        "fl_rounds": 50,  # FL rounds: 50 for MNIST, 3000 for merged mnist, 3500 for CIFAR
        "learning_rate": 0.01,  # SGD optimiser learning rate
        "bsize": 32,  # batch size for training. 10 for merged mnist, 32 for other datasets
        "malicious_clients": 2,  # number of malicious clients (shamir_k = malicious_clients + 1)
        "secure": True,  # enable or disable verification - used for experiments
        "malicious_aggregator": False,  # if True, the aggregator cheat
        "num_colluding_clients": 2,  # must be equal or less than malicious_clients - 0 to disable
        "data_augmentation": False,  # if to augment the data. True only with CIFAR.
        "check_rounding": True,  # if to check the performance with rounding when not using the secure aggregation protocol
        "encoder_base": 4,  # number of decimal places of precision
        "key_size": 1024,  # 515, 1024, and 2048 will use pre-computed values
        "to_log": False,  # if to log the results
        "resume_info": resume_info,  # if to resume from a previous round. Information in the resume_info dictionary
        "save_path": "./",  # path to save the results and models
    }

    assert configuration["num_clients"] >= configuration["clients_participating_per_round"]

    if not configuration["secure"]:
        if (
            configuration["malicious_clients"] != 0
            and configuration["num_colluding_clients"] != 0
            and configuration["malicious_aggregator"]
        ):
            raise ValueError("If not using the secure protocol, then no malicious entities can be supported")

    if configuration["secure"]:
        configuration["save_path"] = os.path.join(
            "experiments", configuration["dataset"], "parallel_secure_" + str(configuration["malicious_clients"])
        )
    else:
        if configuration["check_rounding"]:
            configuration["save_path"] = os.path.join(
                "experiments",
                configuration["dataset"],
                "vanilla_rounded",
                "encoder_base_" + str(configuration["encoder_base"]),
            )
        else:
            configuration["save_path"] = os.path.join("experiments", configuration["dataset"], "vanilla")

    main(configuration)
