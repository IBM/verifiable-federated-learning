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
Script which contains the Encoder class to convert a floating point number into an integer representation.
"""
import math


class Encoder(object):
    """
    Class which handles encoding a floating point weight into an integer representation
    """
    DEFAULT_EXPONENT = 4

    def __init__(self, encoding, exponent):
        """
        Stores the encoding of the weight as well as the exponent used.

        :param encoding: the floating weight encoded in integer format
        :param exponent: the exponent used to encode the weight
        """
        self.encoding = encoding
        self.exponent = exponent

    @classmethod
    def encode(cls, scalar, precision=None):
        """
        Encodes a floating point scalar to a given level of precision

        :param scalar: the weight to encode
        :param precision: the number of decimal places to preserve
        """
        if precision is None:
            encoding = math.floor(round(scalar * pow(10, cls.DEFAULT_EXPONENT)))
            return cls(encoding, cls.DEFAULT_EXPONENT)
        else:
            encoding = math.floor(round(scalar * pow(10, precision)))
            return cls(encoding, precision)
