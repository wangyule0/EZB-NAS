import random

import numpy as np

from genetic.selection import Selection
from utils import GlobalConfigTool, Utils


class Mutation(object):
    def __init__(self, individuals, mutation_prob_, _log):
        self.individuals = individuals
        self.prob = mutation_prob_
        self.log = _log
        self.gen_no = int(individuals[0].id[4:6])

        self.mutation_type_list = GlobalConfigTool.get_mutation_probs()

        self.stage_length_limit = GlobalConfigTool.get_stage_length_limit()
        self.image_channels = GlobalConfigTool.get_input_channels()
        self.downsample_number = GlobalConfigTool.get_downsample_number()
        self.stages_number = self.downsample_number + 1

        self.mobilenet_expansion_factor = GlobalConfigTool.get_mobilenet_expansion_factor()
        self.mobilenet_kernel_size = GlobalConfigTool.get_mobilenet_kernel_size()
        self.mobilenet_se_ratio = GlobalConfigTool.get_mobilenet_se_ratio()
        self.ghostnet_expansion_factor = GlobalConfigTool.get_ghostnet_expansion_factor()
        self.ghostnet_kernel_size = GlobalConfigTool.get_ghostnet_kernel_size()
        self.ghostnet_se_ratio = GlobalConfigTool.get_ghostnet_se_ratio()

        self.max_try_number = GlobalConfigTool.get_max_mutation_number()
        self.selection = Selection()

    def do_mutation(self):
        for indi in self.individuals:
            self.log.info('Mutation occurs in %s' % (indi.id))
            # block mutation
            for stage in indi.stages:
                for unit in stage.units:
                    p_ = random.random()
                    if p_ < self.prob:
                        self.log.info('Mutation occurs at stage %d unit %d' % (stage.id, unit.number))
                        self._do_mutation_unit(stage, unit)
                    else:
                        continue

            indi.reset_fitness()

        Utils.save_individuals_after_mutation(self.individuals, self.gen_no)
        return self.individuals

    def _do_mutation_unit(self, stage, mutation_unit):
        if mutation_unit.number == stage.number_id - 1:
            available_mutation_type = [0, 1, 2, 3]
        else:
            available_mutation_type = [0, 1, 2, 3, 4]  # type, ks, exp, se, filter

        mutation_type = random.choice(available_mutation_type)

        if mutation_type == 0:
            new_type = 2 if mutation_unit.type == 1 else 1
            self.log.info('Stage %d unit %d change type from %d to %d' % (
                stage.id, mutation_unit.number, mutation_unit.type, new_type))
            mutation_unit.type = new_type
        elif mutation_type == 1:
            available_ks_list = [ks for ks in self.mobilenet_kernel_size if ks != mutation_unit.kernel_size]
            new_ks = random.choice(available_ks_list)
            self.log.info('Stage %d unit %d change ks from %d to %d' % (
                stage.id, mutation_unit.number, mutation_unit.kernel_size, new_ks))
            mutation_unit.kernel_size = new_ks
        elif mutation_type == 2:
            available_exp_list = [exp for exp in self.mobilenet_expansion_factor if exp != mutation_unit.expansion_factor]
            new_exp = random.choice(available_exp_list)
            self.log.info('Stage %d unit %d change exp from %d to %d' % (
                stage.id, mutation_unit.number, mutation_unit.expansion_factor, new_exp))
            mutation_unit.expansion_factor = new_exp
        elif mutation_type == 3:
            available_se_list = [exp for exp in self.mobilenet_se_ratio if exp != mutation_unit.se_ratio]
            new_se = random.choice(available_se_list)
            self.log.info('Stage %d unit %d change se from %.2f to %.2f' % (
                stage.id, mutation_unit.number, mutation_unit.se_ratio, new_se))
            mutation_unit.se_ratio = new_se
        elif mutation_type == 4:
            available_out_channels = [channels for channels in stage.out_channels_list if channels != mutation_unit.out_channels]
            new_out = random.choice(available_out_channels)
            self.log.info('Stage %d unit %d change se from %d to %d' % (
                stage.id, mutation_unit.number, mutation_unit.out_channels, new_out))
            mutation_unit.out_channels = new_out
            stage.units[mutation_unit.number + 1].in_channels = new_out
