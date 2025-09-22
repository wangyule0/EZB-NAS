import copy
import hashlib
import random

import numpy as np


class Unit(object):
    def __init__(self, number, in_channels, out_channels):
        self.number = number
        self.in_channels = in_channels
        self.out_channels = out_channels


class MobileUnit(Unit):
    def __init__(self, number, in_channels, out_channels, kernel_size, expansion_factor, se_ratio):
        super().__init__(number, in_channels, out_channels)
        self.type = 1
        self.kernel_size = kernel_size
        self.expansion_factor = expansion_factor
        self.se_ratio = se_ratio


class GhostUnit(Unit):
    def __init__(self, number, in_channels, out_channels, kernel_size, expansion_factor, se_ratio):
        super().__init__(number, in_channels, out_channels)
        self.type = 2
        self.kernel_size = kernel_size
        self.expansion_factor = expansion_factor
        self.se_ratio = se_ratio


class Stage(object):
    def __init__(self, stage_no, params, out_channels_list, in_channels=None, out_channels=None):
        self.id = stage_no  # 0 or 1 or 2
        self.number_id = 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_channels_list = out_channels_list

        self.mobilenet_expansion_factor = params['mobilenet_expansion_factor']
        self.mobilenet_kernel_size = params['mobilenet_kernel_size']
        self.mobilenet_se_ratio = params['mobilenet_se_ratio']
        self.ghostnet_expansion_factor = params['ghostnet_expansion_factor']
        self.ghostnet_kernel_size = params['ghostnet_kernel_size']
        self.ghostnet_se_ratio = params['ghostnet_se_ratio']

        self.min_len = params['stage_length_limit'][self.id][0]
        self.max_len = params['stage_length_limit'][self.id][1]

        self.units = []

    def initialize(self):
        # initialize number of stage
        num_stage_len = np.random.randint(self.min_len, self.max_len + 1)
        unit_in_channels = self.in_channels
        unit_out_channels = None
        # init unit
        for i in range(num_stage_len):
            unit_type = np.random.rand()
            # unit_type = 0.0
            if i == num_stage_len - 1:
                unit_out_channels = self.out_channels
            else:
                unit_out_channels = self.out_channels_list[np.random.randint(0, len(self.out_channels_list))]

            if unit_type < 0.5:
                mobile_unit = self.init_a_mobile_unit(_number=None, _in_channels=unit_in_channels,
                                                      _out_channels=unit_out_channels)
                self.units.append(mobile_unit)
            else:
                ghost_unit = self.init_a_ghost_unit(_number=None, _in_channels=unit_in_channels,
                                                    _out_channels=unit_out_channels)
                self.units.append(ghost_unit)
            unit_in_channels = unit_out_channels

    def init_a_mobile_unit(self, _number, _in_channels, _out_channels, _kernel_size=None, _expansion_factor=None, _se_ratio=None):
        if _number is not None:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1

        if _kernel_size is not None:
            kernel_size = _kernel_size
        else:
            kernel_size = self.mobilenet_kernel_size[np.random.randint(0, len(self.mobilenet_kernel_size))]

        if _expansion_factor is not None:
            expansion_factor = _expansion_factor
        else:
            expansion_factor = self.mobilenet_expansion_factor[np.random.randint(0, len(self.mobilenet_expansion_factor))]

        if _se_ratio is not None:
            se_ratio = _se_ratio
        else:
            se_ratio = self.mobilenet_se_ratio[np.random.randint(0, len(self.mobilenet_se_ratio))]

        mobile_unit = MobileUnit(number, _in_channels, _out_channels, kernel_size, expansion_factor, se_ratio)
        return mobile_unit

    def init_a_ghost_unit(self, _number, _in_channels, _out_channels,  _kernel_size=None, _expansion_factor=None, _se_ratio=None):
        if _number is not None:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1

        if _kernel_size is not None:
            kernel_size = _kernel_size
        else:
            kernel_size = self.mobilenet_kernel_size[np.random.randint(0, len(self.mobilenet_kernel_size))]

        if _expansion_factor is not None:
            expansion_factor = _expansion_factor
        else:
            expansion_factor = self.ghostnet_expansion_factor[np.random.randint(0, len(self.ghostnet_expansion_factor))]

        if _se_ratio is not None:
            se_ratio = _se_ratio
        else:
            se_ratio = self.ghostnet_se_ratio[np.random.randint(0, len(self.ghostnet_se_ratio))]

        ghost_unit = GhostUnit(number, _in_channels, _out_channels, kernel_size, expansion_factor, se_ratio)
        return ghost_unit

    def unit_string(self):
        _str = []
        for unit in self.units:
            _sub_str = []
            if unit.type == 1:
                _sub_str.append('mobilenet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('in:%d' % (unit.in_channels))
                _sub_str.append('out:%d' % (unit.out_channels))
                _sub_str.append('ks:%d' % (unit.kernel_size))
                _sub_str.append('exp:%d' % (unit.expansion_factor))
                _sub_str.append('se:%.2f' % (unit.se_ratio))

            if unit.type == 2:
                _sub_str.append('ghostnet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('in:%d' % (unit.in_channels))
                _sub_str.append('out:%d' % (unit.out_channels))
                _sub_str.append('ks:%d' % (unit.kernel_size))
                _sub_str.append('exp:%d' % (unit.expansion_factor))
                _sub_str.append('se:%.2f' % (unit.se_ratio))

            _str.append('%s%s%s' % ('[', ','.join(_sub_str), ']'))

        return _str


class Individual(object):
    def __init__(self, params, indi_no):
        self.acc = -1.0
        self.params = -1.0
        self.flops = -1.0
        self.zen_score = -1.0
        self.zico_score = -1.0

        self.id = indi_no  # for record the id of current individual    indi%02d%02d
        self.stage_id = 0  # for record the latest number of stage

        self.init_params = params
        self.downsample_number = params['downsample_number']
        self.output_channels = params['output_channels']
        self.init_channels = params['init_channels']
        assert len(self.output_channels) == self.downsample_number + 1, "output channels must = downsample number + 1"

        self.stages = []
        self.downsample_blocks = []

        self.rank = None  # 排名 (Pareto Rank)
        self.crowding_distance = -1.0  # 拥挤度

    def reset_fitness(self):
        self.acc = -1.0
        self.flops = -1.0
        self.params = -1.0
        self.zen_score = -1.0
        self.zico_score = -1.0

    def calculate_zico_bc(self):
        feature_size = [32, 16, 8]
        bc_count = 0.0

        for i, stage in enumerate(self.stages):
            stage_feature_size = feature_size[i]
            for unit in stage.units:
                unit_bc = np.log(stage_feature_size * stage_feature_size / np.sqrt(unit.out_channels))
                bc_count += unit_bc
        return bc_count

    def initialize(self):
        # init stage
        for i, channels_list in enumerate(self.output_channels):
            stage_input_channels = self.init_channels if self.stage_id == 0 else max(self.output_channels[i-1])
            new_stage = Stage(self.stage_id, self.init_params, channels_list, stage_input_channels, max(channels_list))
            new_stage.initialize()
            self.stages.append(new_stage)
            self.stage_id += 1

    def __str__(self):
        _str = []
        _stage_str = []
        # _downsample_str = []
        _str.append('indi:%s' % (self.id))
        _str.append('Acc:%.5f' % (self.acc))
        _str.append('Params:%.5f' % (self.params))
        _str.append('Flops:%.5f' % (self.flops))
        _str.append('Zen:%.5f' % (self.zen_score))
        _str.append('Zico:%.5f' % (self.zico_score))
        # _str.append('Fitness:%.5f' % (self.fitness))

        for stage in self.stages:
            _sub_stage_str = []
            _sub_stage_str.append('Stage%d:' % stage.id)
            _sub_stage_str += stage.unit_string()
            _str.append('%s%s%s' % ('{', '\n'.join(_sub_stage_str), '}'))

        return '\n'.join(_str)


class Population(object):
    def __init__(self, params, gen_no):
        self.gen_no = gen_no
        self.number_id = 0  # for record how many individuals have been generated
        self.pop_size = params['pop_size']
        self.exp_pop_size = params['exp_pop_size']
        self.params = params
        self.individuals = []

    def initialize(self):
        for _ in range(self.pop_size):
            indi_no = 'indi%02d%02d' % (self.gen_no, self.number_id)
            self.number_id += 1
            indi = Individual(self.params, indi_no)
            indi.initialize()
            self.individuals.append(indi)

    def create_from_offspring(self, offsprings):
        for indi_ in offsprings:
            indi = copy.deepcopy(indi_)
            indi_no = 'indi%02d%02d' % (self.gen_no, self.number_id)
            indi.id = indi_no
            self.number_id += 1
            indi.stage_id = len(indi.stages)
            self.individuals.append(indi)

    def __str__(self):
        _str = []
        for ind in self.individuals:
            _str.append(str(ind))
            _str.append('-' * 100)
        return '\n'.join(_str)


def test_stage(params):
    stage = Stage(0, params, 16, 32)
    stage.initialize()

    return stage.unit_string()


def test_individual(params):
    ind = Individual(params, 0)
    ind.initialize()
    print(ind)
    print(ind.uuid())


def test_population(params):
    pop = Population(params, 0)
    pop.initialize()
    print(pop)


if __name__ == '__main__':
    # test_individual(utils.GlobalConfigTool.get_init_params())
    # test_population(utils.GlobalConfigTool.get_init_params())
    # print(test_stage(utils.GlobalConfigTool.get_init_params()))
    pass
