import copy
import random

import numpy as np

from genetic.selection import Selection
from utils import Utils, GlobalConfigTool


class Crossover(object):
    def __init__(self, individuals, crossover_prob_, _log):
        self.individuals = individuals
        self.prob = crossover_prob_
        self.log = _log
        self.gen_no = int(individuals[0].id[4:6])
        self.stage_length_limit = GlobalConfigTool.get_stage_length_limit()
        self.downsample_number = GlobalConfigTool.get_downsample_number()
        self.stages_number = self.downsample_number + 1
        self.image_channels = GlobalConfigTool.get_input_channels()
        self.max_try_number = GlobalConfigTool.get_max_crossover_number()
        self.selection = Selection()

    def do_crossover(self):
        _stat_param = {'offspring_new': 0, 'offspring_from_parent': 0}
        new_offspring_list = []

        self.selection.nsga2_sort(self.individuals, ['zico_score', 'params'], ['zico_score'])

        for _ in range(len(self.individuals) // 2):
            p_ = random.random()

            if p_ < self.prob:
                _stat_param['offspring_new'] += 2
                # find valid parent
                while True:
                    indi1, indi2 = self._choose_two_diff_parents()
                    parent1, parent2 = copy.deepcopy(indi1), copy.deepcopy(indi2)
                    # choose to crossover stage or downsample
                    len1, len2 = self.downsample_number, self.downsample_number
                    for i in range(len(parent1.stages)):
                        len1 += len(parent1.stages[i].units)
                    for i in range(len(parent2.stages)):
                        len2 += len(parent2.stages[i].units)
                    crossover_type = np.random.randint(0, len1 + len2)
                    # generate position
                    selected_stage, pos1, pos2 = self._generate_valid_positions(parent1, parent2)

                    if pos1 is not None and pos2 is not None:
                        break

                # begin crossover
                offspring1, offspring2 = self._crossover_generate_offspring(selected_stage, parent1, pos1, parent2, pos2)
                new_offspring_list.append(offspring1)
                new_offspring_list.append(offspring2)
            else:
                indi1, indi2 = self._choose_two_diff_parents()
                parent1, parent2 = copy.deepcopy(indi1), copy.deepcopy(indi2)
                parent1.reset_fitness()
                parent2.reset_fitness()
                _stat_param['offspring_from_parent'] += 2
                new_offspring_list.append(parent1)
                new_offspring_list.append(parent2)

        self.log.info('Crossover %d offspring are generated, new:%d, others:%d' % (
            len(new_offspring_list), _stat_param['offspring_new'], _stat_param['offspring_from_parent']))

        for i, indi in enumerate(new_offspring_list):
            indi.id = 'indi%02d%02d' % (self.gen_no, i)
        Utils.save_individuals_after_crossover(new_offspring_list, self.gen_no)

        return new_offspring_list

    def _choose_one_parent(self):
        count_ = len(self.individuals)
        idx1 = np.random.randint(0, count_)
        idx2 = np.random.randint(0, count_)
        while idx2 == idx1:
            idx2 = np.random.randint(0, count_)

        return self.selection.nsga2_compare(self.individuals[idx1], self.individuals[idx1])

    def _choose_two_diff_parents(self):
        idx1 = self._choose_one_parent()
        idx2 = self._choose_one_parent()
        while idx2 == idx1:
            idx2 = self._choose_one_parent()

        return idx1, idx2

    # generate valid position
    def _generate_valid_positions(self, parent1, parent2):
        try_count = 0

        while True:
            selected_stage = np.random.randint(0, self.stages_number)
            len1, len2 = len(parent1.stages[selected_stage].units), len(parent2.stages[selected_stage].units)
            pos1, pos2 = np.random.randint(0, len1), np.random.randint(0, len2)
            pos1 = 0
            try_count += 1
            if try_count > self.max_try_number:
                self.log.warn("Crossover try time more than %d in %s and %s" % (self.max_try_number, parent1.id, parent2.id))
                return None, None, None
            self.log.warn('The %d-th try to find the position in %s and %s' % (try_count, parent1.id, parent2.id))

            # make sure len will valid
            stage_len_min, stage_len_max = self.stage_length_limit[selected_stage][0], self.stage_length_limit[selected_stage][1]
            if (pos1 + len2 - pos2 < stage_len_min) or (pos2 + len1 - pos1 > stage_len_max) \
                    or (pos2 + len1 - pos1 < stage_len_min) or (pos1 + len2 - pos2 > stage_len_max):
                self.log.warn('Crossover offspring have invalid length')
                continue
            # if stage0 pos10 pos20
            if selected_stage == 0 and pos1 == 0 and pos2 == 0:
                self.log.warn("Crossover at stage 0 pos1 0 and pos2 0")
                continue

            return selected_stage, pos1, pos2

    # crossover in parents to generate offsprings
    def _crossover_generate_offspring(self, stage_index, parent1, pos1, parent2, pos2):
        # get offspring unit
        unit_list1, unit_list2 = [], []
        stages1 = parent1.stages[stage_index]
        stages2 = parent2.stages[stage_index]
        for i in range(0, pos1):
            unit_list1.append(stages1.units[i])
        for i in range(pos2, len(stages2.units)):
            unit_list1.append(stages2.units[i])

        for i in range(0, pos2):
            unit_list2.append(stages2.units[i])
        for i in range(pos1, len(stages1.units)):
            unit_list2.append(stages1.units[i])

        # reorder the number of each unit based on its order in the list
        for i, unit in enumerate(unit_list1):
            unit.number = i
        for i, unit in enumerate(unit_list2):
            unit.number = i

        self.log.info("Stage:%d, parent1:%s, pos1:%d, parent2:%s, pos2:%d" % (stage_index, parent1.id, pos1, parent2.id, pos2))
        # re-adjust channels
        self.log.info('Re-adjust channels of offspring')
        if pos1 != 0:
            unit_list1[pos1].in_channels = unit_list1[pos1-1].out_channels
        else:
            unit_list1[pos1].in_channels = parent1.stages[stage_index].in_channels
        if pos2 != 0:
            unit_list2[pos2].in_channels = unit_list2[pos2-1].out_channels
        else:
            unit_list2[pos1].in_channels = parent2.stages[stage_index].in_channels

        parent1.stages[stage_index].units = unit_list1
        parent2.stages[stage_index].units = unit_list2
        parent1.stages[stage_index].number_id = len(unit_list1)
        parent2.stages[stage_index].number_id = len(unit_list2)

        stages_list1, stages_list2 = [], []
        for i in range(0, stage_index + 1):
            stages_list1.append(parent1.stages[i])
            stages_list2.append(parent2.stages[i])
        for i in range(stage_index + 1, self.stages_number):
            stages_list1.append(parent2.stages[i])
            stages_list2.append(parent1.stages[i])

        offspring1, offspring2 = parent1, parent2
        offspring1.stages = stages_list1
        offspring2.stages = stages_list2
        offspring1.reset_fitness()
        offspring2.reset_fitness()
        return offspring1, offspring2
