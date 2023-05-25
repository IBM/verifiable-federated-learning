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
sys.path.append('../../')

from commitment.encoder import Encoder


class EncoderTest(unittest.TestCase):

    def test_base_encoding_int(self):
        value = 3
        enc = Encoder.encode(value)
        dec = enc.decode()
        self.assertEqual(value, dec)

    def test_base_encoding_neg_int(self):
        value = -3
        enc = Encoder.encode(value)
        dec = enc.decode()
        self.assertEqual(value, dec)

    def test_base_encoding_float(self):
        value = 3.14
        enc = Encoder.encode(value)
        dec = enc.decode()
        self.assertEqual(value, dec)

    def test_base_encoding_neg_float(self):
        value = -3.14
        enc = Encoder.encode(value)
        dec = enc.decode()
        self.assertEqual(value, dec)

    def test_base_encoding_int_sum(self):
        value_1 = 3
        enc_1 = Encoder.encode(value_1)

        value_2 = 7
        enc_2 = Encoder.encode(value_2)

        enc = Encoder(enc_1.encoding + enc_2.encoding, enc_1.exponent)
        dec = enc.decode()

        self.assertEqual(value_1 + value_2, dec)

    def test_base_encoding_float_sum(self):
        value_1 = 3.14
        enc_1 = Encoder.encode(value_1)

        value_2 = 7.15
        enc_2 = Encoder.encode(value_2)

        enc = Encoder(enc_1.encoding + enc_2.encoding, enc_1.exponent)
        dec = enc.decode()

        self.assertEqual(round(value_1 + value_2, enc.DEFAULT_EXPONENT), dec)

    def test_base_encoding_mix_neg_sum(self):
        value_1 = 3.14
        enc_1 = Encoder.encode(value_1)

        value_2 = -7
        enc_2 = Encoder.encode(value_2)

        enc = Encoder(enc_1.encoding + enc_2.encoding, enc_1.exponent)
        dec = enc.decode()

        self.assertEqual(round(value_1 + value_2, enc.DEFAULT_EXPONENT), dec)

    def test_base_encoding_mix_neg_sum_2(self):
        value_1 = -3.14
        enc_1 = Encoder.encode(value_1)

        value_2 = 7
        enc_2 = Encoder.encode(value_2)

        enc = Encoder(enc_1.encoding + enc_2.encoding, enc_1.exponent)
        dec = enc.decode()

        self.assertEqual(round(value_1 + value_2, enc.DEFAULT_EXPONENT), dec)

