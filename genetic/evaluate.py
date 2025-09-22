import importlib
import os
import sys

import numpy as np

import data_loader
from utils import Utils, GlobalConfigTool


class FitnessEvaluate(object):
    def __init__(self, individuals, log):
        self.individuals = individuals
        self.log = log
        self.image_size = GlobalConfigTool.get_input_size()
        self.batch_size = GlobalConfigTool.get_search_batch_size()

    def generate_to_python_file(self, is_drop_path, no_reslink):
        self.log.info('Begin to generate python files')
        for indi in self.individuals:
            Utils.generate_pytorch_file(indi, is_drop_path, no_reslink)
        self.log.info('Finish the generation of python files')

    def evaluate(self):
        # begin to train
        for indi in self.individuals:
            file_name = indi.id
            module_name = 'scripts.%s' % (file_name)
            if indi.acc is None or indi.acc <= 0:    # need to train
                self.log.info('Begin to train %s' % (file_name))
                _module = importlib.import_module('.', module_name)
                _class = getattr(_module, 'RunModel')
                cls_obj = _class()
                best_acc, params = cls_obj.do_work(file_id=file_name, individual=indi)
                indi.acc, indi.params = best_acc, params
                # delete module
                del sys.modules[module_name]
                os.remove(os.getcwd() + "/scripts/" + file_name + ".py")
            else:   # trained
                self.log.info("%s has inherited the acc, params as %.5f, %.5f" % (indi.id, indi.acc, indi.params))
                os.remove(os.getcwd() + "/scripts/" + file_name + ".py")

        # build after file
        Utils.save_fitness_after_evaluate(self.individuals, int(self.individuals[0].id[4:6]))

    def score(self):
        self.log.info('Begin to compute score')
        data_root = GlobalConfigTool.get_data_root()
        train_batch_size = GlobalConfigTool.get_train_batch_size()
        num_workers = GlobalConfigTool.get_num_workers()
        if GlobalConfigTool.get_dataset() == "cifar10":
            full_train_loader = data_loader.Cifar10.get_train_loader(data_dir=data_root,
                                                                     batch_size=train_batch_size,
                                                                     augment=True,
                                                                     random_seed=None,
                                                                     shuffle=True,
                                                                     num_workers=num_workers,
                                                                     pin_memory=True)
        elif GlobalConfigTool.get_dataset() == "cifar100":
            full_train_loader = data_loader.Cifar100.get_train_loader(data_dir=data_root,
                                                                      batch_size=train_batch_size,
                                                                      augment=True,
                                                                      random_seed=None,
                                                                      shuffle=True,
                                                                      num_workers=num_workers,
                                                                      pin_memory=True)
        else:
            pass

        data_iter = iter(full_train_loader)
        inputs, labels = next(data_iter)

        for indi in self.individuals:
            file_name = indi.id
            module_name = 'scripts.%s' % (file_name)
            if indi.zico_score is None or indi.zico_score <= 0:
                self.log.info('Begin to calculate %s' % (file_name))
                _module = importlib.import_module('.', module_name)
                _class = getattr(_module, 'RunModel')
                cls_obj = _class()
                zen_score, zico_score = cls_obj.score(inputs=inputs, labels=labels, zen_repeat=32)
                indi.zen_score, indi.zico_score = zen_score, zico_score
                # indi.zico_score = zico_score - indi.calculate_zico_bc()
                # delete module
                del sys.modules[module_name]
                os.remove(os.getcwd() + "/scripts/" + file_name + ".py")
            else:   # trained
                self.log.info("%s has inherited the zen, zico, params as %.5f, %.5f, %.5f" % (indi.id, indi.zen_score, indi.zico_score, indi.params))
                os.remove(os.getcwd() + "/scripts/" + file_name + ".py")

        self.log.info('Finish to compute score')

    def params_and_flops(self):
        self.log.info('Begin to compute params and flops')
        for indi in self.individuals:
            file_name = indi.id
            module_name = 'scripts.%s' % (file_name)
            if indi.params is None or indi.params <= 0:
                self.log.info('Begin to calculate %s' % (file_name))
                _module = importlib.import_module('.', module_name)
                _class = getattr(_module, 'RunModel')
                cls_obj = _class()
                params, flops = cls_obj.params_and_flops()
                indi.params = params
                indi.flops = flops
                # delete module
                del sys.modules[module_name]
                os.remove(os.getcwd() + "/scripts/" + file_name + ".py")
            else:  # trained
                self.log.info(
                    "%s has inherited the params, flops as %.5f, %.5f" % (indi.id, indi.params, indi.flops))
                os.remove(os.getcwd() + "/scripts/" + file_name + ".py")

        self.log.info('Finish to compute params and flops')
