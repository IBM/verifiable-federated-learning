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

import sys
import unittest
import datetime
from random import randint, random
from functools import reduce

import torch
from torch import nn
import numpy as np

sys.path.append('../..')

from commitment.params import Params
from commitment.shamir_secret_sharing import ShamirSecretSharing
from commitment.commitment import Commitment

"""
Make a smaller model for testing
"""
class FLModel(nn.Module):

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, num_classes):
        super(FLModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               dilation=(1, 1),
                               padding=(0, 0))
        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2),
                                           stride=2)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               dilation=(1, 1),
                               padding=(0, 0))

        self.fc1 = nn.Linear(in_features=1024,
                             out_features=20)

        self.fc2 = nn.Linear(in_features=20,
                             out_features=num_classes)

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


class CommitmentTest(unittest.TestCase):

    def test_commitment_single_int(self):
        params = Params.test_params()

        w = 3
        r = 11

        com = Commitment.commit(params, w, r)
        self.assertTrue(com.verify(w, r))

    def test_commitment_single_neg_int(self):
        params = Params.test_params()

        w = -3
        r = 11

        com = Commitment.commit(params, w, r)
        self.assertTrue(com.verify(w, r))

    def test_commitment_single_float(self):
        params = Params.test_params()

        w = 3.14
        r = 11

        com = Commitment.commit(params, w, r)
        self.assertTrue(com.verify(w, r))

    def test_commitment_single_neg_float(self):
        params = Params.test_params()

        w = -3.14
        r = 11

        com = Commitment.commit(params, w, r)
        self.assertTrue(com.verify(w, r))

    def test_commitment_int(self):
        params = Params.test_params()

        w_1 = 3
        r_1 = 11
        com_1 = Commitment.commit(params, w_1, r_1)

        w_2 = 7
        r_2 = 22
        com_2 = Commitment.commit(params, w_2, r_2)

        com = com_1 * com_2
        self.assertTrue(com.verify(w_1 + w_2, r_1 + r_2))

    def test_commitment_float(self):
        params = Params.test_params()

        w_1 = 3.14
        r_1 = 11
        com_1 = Commitment.commit(params, w_1, r_1)

        w_2 = 7.15
        r_2 = 22
        com_2 = Commitment.commit(params, w_2, r_2)

        com = com_1 * com_2
        self.assertTrue(com.verify(w_1 + w_2, r_1 + r_2))

    def test_commitment_mix(self):
        params = Params.test_params()

        w_1 = 3.14
        r_1 = 11
        com_1 = Commitment.commit(params, w_1, r_1)

        w_2 = -7
        r_2 = 22
        com_2 = Commitment.commit(params, w_2, r_2)

        com = com_1 * com_2
        self.assertTrue(com.verify(w_1 + w_2, r_1 + r_2))

    def test_commitment_mix_2(self):
        params = Params.test_params()

        w_1 = -3.14
        r_1 = 11
        com_1 = Commitment.commit(params, w_1, r_1)

        w_2 = 7
        r_2 = 22
        com_2 = Commitment.commit(params, w_2, r_2)

        com = com_1 * com_2
        self.assertTrue(com.verify(w_1 + w_2, r_1 + r_2))

    def test_n_commitment_int(self):
        n = 100
        k = 10

        # Trusted Entity
        params = Params.test_params()

        r_i = []
        for _ in range(n):
            r_i.append(randint(1, n**2))

        r_sum = sum(r_i)

        shares = ShamirSecretSharing.generate_shares(r_sum, n, k)

        # Clients
        w_i = []
        com_i = []
        for i in range(n):
            w_i.append(randint(-n**2, n**2))
            com = Commitment.commit(params, w_i[i], r_i[i])
            com_i.append(com)

        # Aggregator
        s = sum(w_i)
        c = reduce((lambda com_1, com_2: com_1 * com_2), com_i)

        # Clients
        secret = ShamirSecretSharing.compute_secret(shares[0:k])
        self.assertEqual(secret, r_sum)
        self.assertTrue(c.verify(s, secret))

    def test_n_commitment_float(self):
        n = 100
        k = 10

        # Trusted Entity
        params = Params.test_params()

        r_i = []
        for _ in range(n):
            r_i.append(randint(1, n ** 2))

        r_sum = sum(r_i)

        shares = ShamirSecretSharing.generate_shares(r_sum, n, k)

        # Clients
        w_i = []
        com_i = []
        for i in range(n):
            w_i.append(round(random(), 4))
            com = Commitment.commit(params, w_i[i], r_i[i])
            com_i.append(com)

        # Aggregator
        s = round(sum(w_i), 4)
        c = reduce((lambda com_1, com_2: com_1 * com_2), com_i)

        # Clients
        secret = ShamirSecretSharing.compute_secret(shares[0:k])
        self.assertEqual(secret, r_sum)

        expected_com = Commitment.commit(params, s, r_sum)
        self.assertTrue(c.verify(s, secret),
                        "c.c: " + str(c.c) + " c.exponent: " + str(c.exponent) +
                        " - expected_com.c: " + str(expected_com.c) +
                        " expected_com.exponent: " + str(expected_com.exponent))

    def test_n_commitment_float_time(self):
        n = 100
        k = 10

        # Trusted Entity
        time_pre = datetime.datetime.now()
        params = Params.test_params()

        r_i = []
        for _ in range(n):
            r_i.append(randint(1, n ** 2))

        r_sum = sum(r_i)

        shares = ShamirSecretSharing.generate_shares(r_sum, n, k)
        time_te = (datetime.datetime.now() - time_pre).total_seconds()

        # Clients
        time_c = []

        w_i = []
        com_i = []
        for i in range(n):
            w_i.append(round(random(), 4))

            time_pre = datetime.datetime.now()
            com = Commitment.commit(params, w_i[i], r_i[i])
            com_i.append(com)
            time_c.append((datetime.datetime.now() - time_pre).total_seconds())

        time_c1 = max(time_c)

        # Aggregator
        time_pre = datetime.datetime.now()
        s = round(sum(w_i), 4)
        c = reduce((lambda com_1, com_2: com_1 * com_2), com_i)
        time_aggr = (datetime.datetime.now() - time_pre).total_seconds()

        # Clients
        time_pre = datetime.datetime.now()
        secret = ShamirSecretSharing.compute_secret(shares[0:k])
        verify = c.verify(s, secret)
        time_c2 = (datetime.datetime.now() - time_pre).total_seconds()

        print("Time added: ", time_te + time_c1 + time_aggr + time_c2)

        self.assertEqual(secret, r_sum)
        self.assertTrue(verify)

    def test_torch_functions(self):

        n = 100
        k = 10

        # Trusted Entity
        params = Params.test_params()

        r_i = []
        for _ in range(n):
            r_i.append(randint(1, n ** 2))

        r_sum = sum(r_i)

        shares = ShamirSecretSharing.generate_shares(r_sum, n, k)

        w_i = []
        com_i = []
        for i in range(n):
            com = []
            weight_matrix = torch.round(torch.rand(size=(100, 100)), decimals=4)
            # convert to vectors for ease of indexing
            weight_matrix = torch.flatten(weight_matrix)
            w_i.append(weight_matrix)
            for val in weight_matrix.detach().numpy().flatten():
                com.append(Commitment.commit(params, val, r_i[i]))
            com_i.append(com)

        s = torch.stack(w_i)
        s = torch.round(torch.sum(s, dim=0), decimals=4)
        s = s.detach().numpy()

        secret = ShamirSecretSharing.compute_secret(shares[0:k])
        self.assertEqual(secret, r_sum)
        expected_com = []
        for si in s:
            expected_com.append(Commitment.commit(params, si, r_sum))

        for element in range(len(s)):
            mul = com_i[0][element]
            for client in range(1, n):
                mul = mul * com_i[client][element]

            element_s = s[element]
            self.assertTrue(mul.verify(element_s, secret),
                            "mul.c: " + str(mul.c) + " mul.exponent: " + str(mul.exponent) +
                            " - expected_com.c: " + str(expected_com[element].c) +
                            " expected_com.exponent: " + str(expected_com[element].exponent))

    def test_fl_functions(self):
        import copy

        from main import round_tensor_values
        from fl.aggregator import Aggregator
        from fl.data import setup_mnist_clients

        n = 5
        k = 4
        # Trusted Entity
        params = Params.test_params()

        r_i = []
        for _ in range(n):
            r_i.append(randint(1, n ** 2))

        r_sum = sum(r_i)

        shares = ShamirSecretSharing.generate_shares(r_sum, n, k)

        list_of_clients = setup_mnist_clients(config_dic={'num_clients': n,
                                                          'bsize': 32,
                                                          'dataset': 'mnist',
                                                          'encoder_base': 4,
                                                          'key_size': -1})
        global_model = FLModel(10)
        client_models = []
        client_com = []

        for _ in range(n):
            client_models.append(copy.deepcopy(FLModel(10).state_dict()))

        for i in range(len(client_models)):
            client_models[i] = round_tensor_values(client_models[i], config_dic={'encoder_base': 4})

        for i in range(len(client_models)):
            com, _ = list_of_clients[i].compute_commitments(params,
                                                            client_models[i],
                                                            r_i[i])
            client_com.append(com)

        agg = Aggregator()

        # compute sums
        model, com_aggregator_model, aggregator_time = agg.secure_fed_sum(malicious_aggregator=False,
                                                                          global_model=global_model,
                                                                          client_models=client_models,
                                                                          com_model_clients=client_com,
                                                                          config_dic={'check_rounding': True,
                                                                                       'secure': True,
                                                                                       'encoder_base': 4})

        secret = ShamirSecretSharing.compute_secret(shares[0:k])
        self.assertEqual(secret, r_sum)
        for i in range(len(client_models)):
            check, _ = list_of_clients[i].verify_commitments(shares,
                                                             model.state_dict(),
                                                             com_aggregator_model)
            if not check:
                print("Aggregator cheat!")

    # @unittest.skip("demonstrating skipping")
    def test_fl_loop(self):
        import copy

        from main import round_tensor_values
        from fl.aggregator import Aggregator
        from fl.data import setup_mnist_clients
        from torch import optim

        n = 5
        k = 4
        # Trusted Entity
        params = Params.test_params()

        r_i = []
        for _ in range(n):
            r_i.append(randint(1, n ** 2))

        r_sum = sum(r_i)

        shares = ShamirSecretSharing.generate_shares(r_sum, n, k)

        list_of_clients = setup_mnist_clients(config_dic={'dataset': 'mnist',
                                                          'num_clients': n,
                                                          'bsize': 128,
                                                          'encoder_base': 4,
                                                          'key_size': -1})
        model = FLModel(10)
        opt = optim.SGD(model.parameters(), lr=0.01)

        initial_state_dic = copy.deepcopy(model.state_dict())
        initial_opt_state_dic = copy.deepcopy(opt.state_dict())

        client_models = []
        client_com = []

        for i in range(n):
            model.load_state_dict(copy.deepcopy(initial_state_dic))
            opt.load_state_dict(copy.deepcopy(initial_opt_state_dic))
            model, opt, _, _ = list_of_clients[i].train_loop(model=model, opt=opt)

            client_models.append(copy.deepcopy(model.state_dict()))

            # compute commitment
            client_models[-1] = round_tensor_values(client_models[-1], config_dic={'encoder_base': 4})
            print(f'Computing commitment for client number {i}')
            com, _ = list_of_clients[i].compute_commitments(params, client_models[-1], r_i[i])
            client_com.append(com)


        agg = Aggregator()
        # compute sums
        model, com_aggregator_model, _ = agg.secure_fed_sum(malicious_aggregator=False,
                                                            global_model=model,
                                                            client_models=client_models,
                                                            com_model_clients=client_com,
                                                            config_dic={'check_rounding': True,
                                                                        'secure': True,
                                                                        'encoder_base': 4})

        secret = ShamirSecretSharing.compute_secret(shares[0:k])
        self.assertEqual(secret, r_sum)
        for i in range(len(client_models)):
            check, _ = list_of_clients[i].verify_commitments(shares,
                                                             model.state_dict(),
                                                             com_aggregator_model)
            if not check:
                print("Aggregator cheat!")

    def test_all_fl_loop(self):

        print('%---------------------------%')
        print('starting test test_all_fl_loop')
        print('%---------------------------%')

        import copy
        from main import round_tensor_values, init, average_summed_values
        from fl.aggregator import Aggregator
        from torch import optim

        config_dic = {
            'dataset': 'mnist',
            'num_clients': 5,  # not applicable for femnist
            'clients_participating_per_round': 5,
            'fl_rounds': 3,
            'learning_rate': 0.001,
            'bsize': 32,
            'malicious_clients': 1,  # number of malicius clients (shamir_k = clients_participating_per_round - malicious_clients)
            'secure': True,  # enable or disable verification - used for experiments
            'malicious_aggregator': False,  # if True, the aggregator cheat
            'num_colluding_clients': 1,  # must be equal or less than malicious_clients - 0 to disable
            'encoder_base': 4,
            'key_size': -1,
            'resume_info': {'to_resume': False}
        }
        # Trusted Entity
        params, shamir_k, r_i_rounds, shares_rounds, list_of_clients, _, _ = init(config_dic)

        model = FLModel(10)
        opt = optim.SGD(model.parameters(), lr=0.01)

        for round_num in range(config_dic['fl_rounds']):

            initial_state_dic = copy.deepcopy(model.state_dict())
            initial_opt_state_dic = copy.deepcopy(opt.state_dict())
            client_models = []
            client_com = []

            for i in range(config_dic['num_clients']):
                model.load_state_dict(copy.deepcopy(initial_state_dic))
                opt.load_state_dict(copy.deepcopy(initial_opt_state_dic))
                model, opt, _, _ = list_of_clients[i].train_loop(model=model, opt=opt)

                client_models.append(copy.deepcopy(model.state_dict()))

                # compute commitment
                client_models[-1] = round_tensor_values(client_models[-1], config_dic={'encoder_base': 4})
                print(f'Computing commitment for client number {i}')
                com, _ = list_of_clients[i].compute_commitments(params, client_models[-1], r_i_rounds[round_num][i])
                client_com.append(com)


            agg = Aggregator()
            # compute sums
            model, com_aggregator_model, _ = agg.secure_fed_sum(malicious_aggregator=False,
                                                                global_model=model,
                                                                client_models=client_models,
                                                                com_model_clients=client_com,
                                                                config_dic=config_dic)

            shares = shares_rounds[round_num][0:shamir_k]

            # secret = ShamirSecretSharing.compute_secret(shares_rounds[round_num][0:config_dic['num_colluding_clients']])
            secret = ShamirSecretSharing.compute_secret(shares_rounds[round_num][0:shamir_k])
            self.assertEqual(secret, sum(r_i_rounds[round_num]))

            for i in range(len(client_models)):
                check, _ = list_of_clients[i].verify_commitments(shares,
                                                                 model.state_dict(),
                                                                 com_aggregator_model)
                if not check:
                    print("Aggregator cheat!")
                else:
                    print(f"Round {round_num} verified")

            model = average_summed_values(model=model,
                                          clients_participating_per_round=config_dic['clients_participating_per_round'])
