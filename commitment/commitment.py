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
Script with the Commitment class to compute and verify the commitments
"""
from __future__ import annotations

import numpy as np
from commitment.encoder import Encoder
from typing import TYPE_CHECKING, Optional, TypeVar
Self = TypeVar("Self", bound="Commitment")

if TYPE_CHECKING:
    from type_utils import Params


class Commitment(object):
    """
    Commitment class to compute and verify the commitments
    """
    def __init__(self, params: "Params", c=Optional[Self], exponent=None) -> None:
        """
        Setup for the commitment

        :param params: A Params object containing public parameters g, h, and q
        :param c: The commitment
        :param exponent: exponent used to encode floating point numbers
        """
        self.params = params
        self.c = c or 0
        self.exponent = exponent or Encoder.DEFAULT_EXPONENT

    @staticmethod
    def commit(params: "Params", w: np.float32, r: int) -> Commitment:
        """
        Produce the commitment of the value w based on the public parameters and the random number r.

        :params w: value to compute the commitment of, generally one of the weights in the neural network.
        :params r: random number held by the client
        :return: Commitment object generated with the supplied parameters
        """
        w_enc = Encoder.encode(w)

        g_w = pow(params.g, w_enc.encoding, params.q)
        h_r = pow(params.h, r, params.q)

        c = (g_w * h_r) % params.q
        return Commitment(params, c, w_enc.exponent)

    def verify(self, w: np.float32, r: int) -> bool:
        """
        Verify that the commitment from the aggregated model matches

        :params w: value to compute the commitment of, generally one of the weights in the neural network.
        :params r: random number from shamir secret sharing
        :return: True/False if the commitment was verified
        """
        com = Commitment.commit(self.params, w, r)
        check_c = self.c == com.c
        check_exponent = self.exponent == com.exponent
        return check_c & check_exponent

    def __mul__(self, c: Self) -> Commitment:
        """
        Performs multiplication of two commitments

        :param c: Commitment to multiply with
        :return: Commitment object resulting from the multiplication of commitments
        """
        c_mul = (self.c * c.c) % self.params.q
        return Commitment(self.params, c_mul, self.exponent)
