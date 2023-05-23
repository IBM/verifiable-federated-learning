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
Script with the Setup class to perform the cryptographic setup for the protocol
"""
from random import randint
from typing import Tuple, List

from commitment.params import Params
from commitment.shamir_secret_sharing import ShamirSecretSharing


class Setup(object):
    """
    Class to perform the cryptographic setup for the protocol
    """

    @classmethod
    def setup(cls, key_size: int, n: int, k: int, fl_rounds: int) -> Tuple[Params, List, List[List[Tuple[int, bytes]]]]:
        """
        :param key_size: Key length used for the commitments
        :param n: The number of clients participating per round
        :param k: The number of clients that are needed to reconstruct the secret
        :param fl_rounds: Number of FL rounds to run. We pre-compute the r's and the shares here.
        :return: A Tuple containing
                    1) A Params object with public parameters g, h, and q as attributes.
                    2) A list with the (private) random number per client per round
                    3) A list with the SSS shares per round.
        """
        if key_size == 2048:
            params = Params.test_params_2048()
        elif key_size == 1024:
            params = Params.test_params_1024()
        elif key_size == 512:
            params = Params.test_params_512()
        elif key_size == -1:
            params = Params.test_params()
        else:
            print("Generating new parameters")
            params = Params.generate_params(n_length=key_size)

        r_i_rounds = []
        shares_rounds = []
        for _ in range(fl_rounds):
            r_i, secret = cls.generate_random(n)
            r_i_rounds.append(r_i)
            shares = cls.generate_shares(secret, n, k)
            shares_rounds.append(shares)

        return params, r_i_rounds, shares_rounds

    @staticmethod
    def generate_random(n: int) -> Tuple[List[int], int]:
        """
        Generates the random r per client and computes the secret sum of r

        :param n: number of clients participating in a FL round
        :return: the random number per client and their sum
        """
        r_i = []
        for _ in range(n):
            r_i.append(randint(1, n**2))

        r_sum = sum(r_i)
        return r_i, r_sum

    @staticmethod
    def generate_shares(secret: int, n: int, k: int) -> List[Tuple[int, bytes]]:
        """
        Using the Shamir Secret Sharing Protocol split the secret into shares

        :param secret: the secret to split, in our case the sum of the private r
        :param n: number of clients in the round
        :param k: number of shares requires to re-construct the secret.
        :return: Shares from the Shamir Secret Sharing Protocol
        """
        return ShamirSecretSharing.generate_shares(secret, n, k)
