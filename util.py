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
Script which contains a utility class to help with logging results and loading/saving models
"""
import os
import datetime
import json
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from main import ConfigDict


class Logger:
    """
    Utility class to help with logging experimental results and loading/saving models
    """
    def __init__(self, config_dic: "ConfigDict"):
        self.model_savepath = config_dic['save_path']

        if not os.path.isdir(self.model_savepath):
            os.makedirs(self.model_savepath)

        if not config_dic['resume_info']['to_resume']:
            self.index = 0
            self.save_folder = f'run_{str(self.index)}'

            while os.path.isdir(os.path.join(self.model_savepath, self.save_folder)):
                self.index += 1
                self.save_folder = 'run_' + str(self.index)
        else:
            if config_dic['resume_info']['resume_run'] is not None:
                self.save_folder = config_dic['resume_info']['resume_run']
                self.index = 0
                self.save_folder = os.path.join(config_dic['resume_info']['resume_run'], 'resume_' + str(self.index))
                while os.path.isdir(os.path.join(self.model_savepath, self.save_folder)):
                    self.index += 1
                    self.save_folder = os.path.join(config_dic['resume_info']['resume_run'], 'resume_' + str(self.index))
            else:
                raise ValueError("resume_run is None in the resume info")

        self.model_savepath = os.path.join(self.model_savepath, self.save_folder)
        os.makedirs(self.model_savepath)
        print('Experiments will be saved to: ', self.model_savepath, flush=True)
        with open(os.path.join(self.model_savepath, 'fl_configuration.json'), 'w') as config_file:
            json.dump(config_dic, config_file, sort_keys=True, indent=4)

        if not os.path.isdir(os.path.join(self.model_savepath, 'models')):
            os.makedirs(os.path.join(self.model_savepath, 'models'))
        self.pytorch_model_savepath = os.path.join(self.model_savepath, 'models')

        with open(os.path.join(self.model_savepath, 'results.csv'), 'a') as f_open:
            f_open.write(','.join(list(map(str, ["round_num", "test_loss", "test_acc"]))) + '\n')

        with open(os.path.join(self.model_savepath, 'round_times.csv'), 'a') as f_open:
            f_open.write(','.join(list(map(str, ["round_num", "round_time"]))) + '\n')

    def log_results(self, info, file_name: str = 'results.csv') -> None:
        """
        Logs the training/test results for loss and accuracy

        :param info: experiment results to save
        :param file_name: name of the file to write to
        :return: None
        """
        info = ','.join(info) + '\n'
        with open(os.path.join(self.model_savepath, file_name), 'a') as f_open:
            f_open.write(info)

    def log_times(self, times, file_name: str) -> None:
        """
        Log times. For clients time this is a nested dict, of Dict[round_number][client_index] = times.
        For aggregator this is a Dict of Dict[round_number] = times
        :param times:
        :param file_name:

        :return: None
        """
        with open(os.path.join(self.model_savepath, file_name), 'w') as f_open:
            json.dump(times, f_open, indent=4)

    def fetch_model_resume_file(self, config_dic: "ConfigDict") -> str:
        """
        Finds the most up-to-date model save dictionary if resuming a training round.

        :param config_dic: Dictionary defining the configuration of the experiment
        :return: path to pytorch state dict
        """
        resume_num = os.path.normpath(self.save_folder).split(os.path.sep)[-1]
        if config_dic['resume_info']['resume_run'] is None:
            raise ValueError("resume_run is None in the resume info")
        if resume_num == 'resume_0':  # first resumption, go to top level
            model_path = os.path.join(config_dic['save_path'],
                                      config_dic['resume_info']['resume_run'],
                                      'models', 'checkpoint_di')
        elif 'resume_' in resume_num:  # Nth resumption, go to prior resumption folder
            tmp = int(resume_num[-1]) - 1
            model_path = os.path.join(
                config_dic['save_path'],
                config_dic['resume_info']['resume_run'],
                f'resume_{str(tmp)}',
                'models',
                'checkpoint_dict',
            )
        else:
            raise ValueError('Model loading path not parsed correctly')
        return model_path

    def save_models(self, model: torch.nn.Module, opt: torch.optim.Optimizer) -> None:
        """
        Save a pytorch model and optimizer

        :param model: Pytorch model to save
        :param opt: Pytorch optimizer to save
        :return: None
        """
        torch.save({'model_head': model.state_dict(),
                    'opt': opt.state_dict()},
                   os.path.join(self.pytorch_model_savepath, 'checkpoint_dict'))

    def save_time(self, experiment: str, time_pre) -> None:
        """
        Saves the experiment duration
        :param experiment: experiment name
        :param time_pre: start time
        """
        time_post = datetime.datetime.now()
        self.write_csv(experiment, str((time_post - time_pre).total_seconds()))

    def write_csv(self, entity, row) -> None:
        """
        Write results to csv file
        :param entity: file name
        :param row: data to save
        """
        with open(os.path.join(self.model_savepath, f'{entity}_time.csv'), 'a') as f_open:
            f_open.write(row + '\n')
