import copy
import os
import random

import numpy as np

from genetic.crossover import Crossover
from genetic.evaluate import FitnessEvaluate
from genetic.mutation import Mutation
from genetic.population import Population, Individual
from genetic.selection import Selection
from utils import GlobalConfigTool, Log, Utils


class EvolveCNN(object):
    def __init__(self, _params, _log):
        self.params_min, self.params_max = GlobalConfigTool.get_params_limit()
        self.params = _params
        self.log = _log
        self.gen_no = None
        self.pop = None
        self.parent_pop = None
        self.pop_size = GlobalConfigTool.get_pop_size()
        self.selection = Selection()

    def initialize_population(self):
        GlobalConfigTool.begin_evolution()
        self.gen_no = 0
        new_pop = Population(self.params, self.gen_no)
        new_pop.initialize()
        self.pop = new_pop

    def load_population(self):
        gen_no_ = Utils.get_newest_file_based_on_prefix('begin')
        if gen_no_ is not None:
            self.log.info('Initialize from %d-th generation' % (gen_no_))
            pop_ = Utils.load_population('begin', gen_no_)
            self.gen_no = gen_no_
            self.pop = pop_
        else:
            raise ValueError('The running flag is set to be running, but there is no generated population stored')

    def load_evolved_population(self):
        gen_no_ = Utils.get_newest_file_based_on_prefix('begin')
        if gen_no_ is not None:
            self.log.info('Loading mutation population from %d-th generation' % (gen_no_))
            pop_ = Utils.load_population('offspring', gen_no_)
            parent_pop_ = Utils.load_population('begin', gen_no_)
            self.gen_no = gen_no_
            self.pop = pop_
            self.parent_pop = parent_pop_
            # load acc, params from train log
            for indi in self.pop.individuals:
                acc, params, flops, zen_score, zico_score = Utils.read_log_fitness(indi.id)
                indi.acc, indi.params, indi.flops, indi.zen_score, indi.zico_score = acc, params, flops, zen_score, zico_score
        else:
            raise ValueError('The evolved flag is set to be evolved, but there is no generated population stored')

    def fitness_evaluate(self):
        fitness = FitnessEvaluate(self.pop.individuals, self.log)
        fitness.generate_to_python_file(is_drop_path=False, no_reslink=True)
        fitness.score()
        fitness.generate_to_python_file(is_drop_path=False, no_reslink=False)
        fitness.params_and_flops()

        Utils.save_score(self.pop.individuals, int(self.pop.individuals[0].id[4:6]))
        # fitness.evaluate(is_test=False)

    def final_train(self, indis):
        fitness = FitnessEvaluate(indis, self.log)
        fitness.generate_to_python_file(is_drop_path=True, no_reslink=False)
        fitness.evaluate()

    def crossover(self):
        crossover_ = Crossover(self.pop.individuals, self.params['prob_crossover'], self.log)
        offspring = crossover_.do_crossover()
        self.pop.individuals = offspring

    def mutation(self):
        mutation_ = Mutation(self.pop.individuals, self.params['prob_mutation'], self.log)
        offspring = mutation_.do_mutation()
        self.pop.individuals = offspring

    def environment_selection(self):
        available_individuals = self.pop.individuals + self.parent_pop.individuals
        next_individuals = self.selection.nsga2_selection(available_individuals, self.params['pop_size'], ['zico_score', 'params'], ['zico_score'])

        _str = []
        for _, indi in enumerate(self.pop.individuals):
            _f_str = 'Indi-%s-%.5f-%.5f-%d-%.5f' % (indi.id, indi.acc, indi.params, indi.rank, indi.crowding_distance)
            _str.append(_f_str)
        for _, indi in enumerate(self.parent_pop.individuals):
            _f_str = 'Pare-%s-%.5f-%.5f-%d-%.5f' % (indi.id, indi.acc, indi.params, indi.rank, indi.crowding_distance)
            _str.append(_f_str)

        next_population = Population(self.params, self.gen_no + 1)
        next_population.create_from_offspring(next_individuals)
        self.pop = next_population

        for _, indi in enumerate(self.pop.individuals):
            _t_str = 'New-%s-%.5f-%.5f-%d-%.5f' % (indi.id, indi.acc, indi.params, indi.rank, indi.crowding_distance)
            _str.append(_t_str)
        _file = './populations/envi_%02d.txt' % (self.gen_no)
        Utils.write_to_file('\n'.join(_str), _file)

    def do_work(self):
        self.log.important('*' * 25)
        self.log.important('EVOLVE-begin Initialize')

        running_flag = GlobalConfigTool.is_evolution_running()
        evolved_flag = GlobalConfigTool.is_evolution_evolved()

        if (not running_flag) and evolved_flag:
            raise ValueError("Running flag is false, but evolved flag is true.")

        if running_flag:
            self.log.info('Initialize from existing population data')
            if not evolved_flag:
                self.load_population()
            else:
                self.load_evolved_population()
        else:
            self.log.info('Initialize new population')
            self.initialize_population()
        self.log.important('EVOLVE-finish Initialize')

        if not running_flag:
            # new population need to evaluation
            self.log.important('EVOLVE[%d-gen]-begin the Evaluation' % (self.gen_no))
            self.fitness_evaluate()
            self.log.important('EVOLVE[%d-gen]-finish the Evaluation' % (self.gen_no))

        begin_gen_no = self.gen_no
        for curr_gen in range(begin_gen_no, self.params['max_gen']):
            # only the first generation and is running, do not need to save population
            if not (curr_gen == begin_gen_no and running_flag):
                self.log.important('EVOLVE[%d-gen]-save the begin population' % (self.gen_no))
                Utils.save_individuals_at_begin(self.pop.individuals, self.gen_no)

            # only the first generation and is evolved, do not need to copy parent, crossover, mutation
            if not (curr_gen == begin_gen_no and evolved_flag):
                self.parent_pop = copy.deepcopy(self.pop)

                self.log.important('EVOLVE[%d-gen]-begin the Crossover' % (curr_gen))
                self.crossover()
                self.log.important('EVOLVE[%d-gen]-finish the Crossover' % (curr_gen))

                self.log.important('EVOLVE[%d-gen]-begin the Mutation' % (curr_gen))
                self.mutation()
                self.log.important('EVOLVE[%d-gen]-finish the Mutation' % (curr_gen))

            self.log.important('EVOLVE[%d-gen]-begin the Evaluation' % (curr_gen))
            self.fitness_evaluate()
            self.log.important('EVOLVE[%d-gen]-finish the Evaluation' % (curr_gen))

            self.log.important('EVOLVE[%d-gen]-begin the Environment Selection' % (curr_gen))
            self.environment_selection()
            self.log.important('EVOLVE[%d-gen]-finish the Environment Selection' % (curr_gen))

            self.gen_no += 1

        Utils.save_individuals_at_begin(self.pop.individuals, self.gen_no)  # save final population

        self.log.important('EVOLVE[%d-gen]-begin the Final Train' % (self.gen_no))
        self.final_train(self.pop.individuals)
        self.log.important('EVOLVE[%d-gen]-finish the Final Train' % (self.gen_no))
        GlobalConfigTool.end_evolution()


def final():
    # sort by params
    gen_no = Utils.get_newest_file_based_on_prefix('begin')
    pop = Utils.load_population('begin', gen_no)

    selection = Selection()
    fronts = selection.fast_non_dominated_sort(pop.individuals, ['zico_score', 'params'], ['zico_score'])

    sort_individuals = sorted(fronts[0], key=lambda indi: indi.params)

    sort_pop = Population(GlobalConfigTool.get_init_params(), gen_no)
    sort_pop.create_from_offspring(sort_individuals)
    for indi in sort_pop.individuals:
        indi.params = -1.0

    fitness = FitnessEvaluate(sort_pop.individuals, Log)
    fitness.generate_to_python_file(is_drop_path=True, no_reslink=False)
    fitness.params_and_flops()

    Utils.save_individuals_at_begin(sort_pop.individuals, gen_no)

def test():
    pop = Utils.load_population('begin', 90)
    fitness = FitnessEvaluate(pop.individuals, Log)
    fitness.generate_to_python_file(is_drop_path=False, no_reslink=False)


if __name__ == '__main__':
    params_ = GlobalConfigTool.get_init_params()
    evolve_cnn = EvolveCNN(params_, Log)
    evolve_cnn.do_work()
    final()
    # test()
