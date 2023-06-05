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
Script to handle the Shamir Secret Sharing protocol
"""
from typing import List, Tuple

from Crypto.Protocol.SecretSharing import Shamir


class ShamirSecretSharing(object):
    """
    Class which handles the Shamir Secret Sharing protocol
    """
    @staticmethod
    def generate_shares(secret: int, n: int, k: int) -> List[Tuple[int, bytes]]:
        """
        Generate the shares to split the secret

        :param secret: value to hide
        :param n: Total number of clients participating in a round
        :param k: Number of shares required to recompute the secret.
                  Needs to be greater than the number of malicious clients
        """
        secret = secret.to_bytes(16, "big")
        return Shamir.split(k, n, secret, False)

    @staticmethod
    def compute_secret(shares: List[Tuple[int, bytes]]) -> int:
        """
        Recomputes the secret by combining the shares
        :param shares: shares to use for recomputing the secret
        :return: the secret
        """
        secret = Shamir.combine(shares, False)
        return int.from_bytes(secret, 'big')
