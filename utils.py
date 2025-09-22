import configparser
import logging
import os
import sys
import time

import numpy as np

from genetic.population import Individual, Population, Stage


class GlobalConfigTool(object):
    global_config_path = r'global.ini'

    @classmethod
    def clear_config(cls):
        config = configparser.ConfigParser()
        config.read(cls.global_config_path)
        secs = config.sections()
        for sec_name in secs:
            item_list = config.options(sec_name)
            for item_name in item_list:
                config.set(sec_name, item_name, " ")
        config.write(open(cls.global_config_path, 'w'))

    @classmethod
    def __write_ini_file(cls, section, key, value):
        config = configparser.ConfigParser()
        config.read(cls.global_config_path)
        config.set(section, key, value)
        config.write(open(cls.global_config_path, 'w'))

    @classmethod
    def __read_ini_file(cls, section, key):
        config = configparser.ConfigParser()
        config.read(cls.global_config_path)
        return config.get(section, key)

    # [evolution_status]
    @classmethod
    def begin_evolution(cls):
        section = 'evolution_status'
        key = 'IS_RUNNING'
        cls.__write_ini_file(section, key, "1")

    @classmethod
    def end_evolution(cls):
        section = 'evolution_status'
        key = 'IS_RUNNING'
        cls.__write_ini_file(section, key, "0")

    @classmethod
    def is_evolution_running(cls):
        rs = cls.__read_ini_file('evolution_status', 'IS_RUNNING')
        if rs == '1':
            return True
        else:
            return False

    @classmethod
    def is_evolution_evolved(cls):
        rs = cls.__read_ini_file('evolution_status', 'IS_EVOLVED')
        if rs == '1':
            return True
        else:
            return False

    # [evolution_settings]
    @classmethod
    def get_pop_size(cls):
        rs = cls.__read_ini_file('evolution_settings', 'pop_size')
        return int(rs)

    @classmethod
    def get_max_generation(cls):
        rs = cls.__read_ini_file('evolution_settings', 'max_generation')
        return int(rs)

    @classmethod
    def get_genetic_probability(cls):
        rs = cls.__read_ini_file('evolution_settings', 'genetic_prob').split(',')
        p = [float(i) for i in rs]
        return p

    @classmethod
    def get_mutation_probs(cls):
        rs = cls.__read_ini_file('evolution_settings', 'mutation_probs').split(',')
        assert len(rs) == 3
        mutation_prob_list = [float(i) for i in rs]
        return mutation_prob_list

    @classmethod
    def get_max_crossover_number(cls):
        rs = cls.__read_ini_file('evolution_settings', 'max_crossover_number')
        return int(rs)

    @classmethod
    def get_max_mutation_number(cls):
        rs = cls.__read_ini_file('evolution_settings', 'max_mutation_number')
        return int(rs)

    @classmethod
    def get_params_limit(cls):
        rs = cls.__read_ini_file('evolution_settings', 'params_limit').split(',')
        params_limit_list = [float(i) for i in rs]
        return params_limit_list

    @classmethod
    def get_params_weight(cls):
        rs = cls.__read_ini_file('evolution_settings', 'params_weight').split(',')
        params_weight_list = [float(i) for i in rs]
        return params_weight_list

    # [network]
    @classmethod
    def get_stage_length_limit(cls):
        rs = cls.__read_ini_file('network', 'stage_length_limit')
        length_limit = []
        for stage_limit in rs.split(';'):
            stage = []
            for i in stage_limit.split(','):
                stage.append(int(i))
            length_limit.append(stage)
        return length_limit

    @classmethod
    def get_downsample_number(cls):
        rs = cls.__read_ini_file('network', 'downsample_number')
        return int(rs)

    @classmethod
    def get_input_size(cls):
        rs = cls.__read_ini_file('network', 'input_size')
        return int(rs)

    @classmethod
    def get_input_channels(cls):
        rs = cls.__read_ini_file('network', 'input_channels')
        return int(rs)

    @classmethod
    def get_init_channels(cls):
        rs = cls.__read_ini_file('network', 'init_channels')
        return int(rs)

    @classmethod
    def get_output_channels(cls):
        rs = cls.__read_ini_file('network', 'output_channels')
        result = []
        for sub_str in rs.split(';'):
            channels = []
            for i in sub_str.split(','):
                channels.append(int(i))
            result.append(channels)
        return result

    @classmethod
    def get_num_class(cls):
        rs = cls.__read_ini_file('network', 'num_class')
        return int(rs)

    @classmethod
    def get_drop_path_rate(cls):
        rs = cls.__read_ini_file('network', 'drop_path_rate')
        return float(rs)

    # [search]
    @classmethod
    def get_search_epoch(cls):
        rs = cls.__read_ini_file('search', 'search_epoch')
        return int(rs)

    @classmethod
    def get_final_epoch(cls):
        rs = cls.__read_ini_file('search', 'final_epoch')
        return int(rs)

    @classmethod
    def get_dataset(cls):
        rs = cls.__read_ini_file('search', 'dataset')
        return str(rs)

    @classmethod
    def get_data_root(cls):
        rs = cls.__read_ini_file('search', 'data_root')
        return str(rs)

    @classmethod
    def get_search_batch_size(cls):
        rs = cls.__read_ini_file('search', 'search_batch_size')
        return int(rs)

    @classmethod
    def get_train_batch_size(cls):
        rs = cls.__read_ini_file('search', 'train_batch_size')
        return int(rs)

    @classmethod
    def get_num_workers(cls):
        rs = cls.__read_ini_file('search', 'num_workers')
        return int(rs)

    @classmethod
    def get_weight_decay(cls):
        rs = cls.__read_ini_file('search', 'weight_decay')
        return float(rs)

    @classmethod
    def get_gpu(cls):
        rs = cls.__read_ini_file('search', 'gpu')
        return int(rs)

    @classmethod
    def get_random_seed(cls):
        rs = cls.__read_ini_file('search', 'random_seed')
        return int(rs)

    # [mobilenet_configuration]
    @classmethod
    def get_mobilenet_expansion_factor(cls):
        rs = cls.__read_ini_file('mobilenet_configuration', 'expansion_factor')
        expansion_factor = []
        for i in rs.split(','):
            expansion_factor.append(int(i))
        return expansion_factor

    @classmethod
    def get_mobilenet_kernel_size(cls):
        rs = cls.__read_ini_file('mobilenet_configuration', 'kernel_size')
        kernel_size = []
        for i in rs.split(','):
            kernel_size.append(int(i))
        return kernel_size

    @classmethod
    def get_mobilenet_se_ratio(cls):
        rs = cls.__read_ini_file('mobilenet_configuration', 'se_ratio')
        se_ratio = []
        for i in rs.split(','):
            se_ratio.append(float(i))
        return se_ratio

    # [ghostnet_configuration]
    @classmethod
    def get_ghostnet_expansion_factor(cls):
        rs = cls.__read_ini_file('ghostnet_configuration', 'expansion_factor')
        expansion_factor = []
        for i in rs.split(','):
            expansion_factor.append(int(i))
        return expansion_factor

    @classmethod
    def get_ghostnet_kernel_size(cls):
        rs = cls.__read_ini_file('ghostnet_configuration', 'kernel_size')
        kernel_size = []
        for i in rs.split(','):
            kernel_size.append(int(i))
        return kernel_size

    @classmethod
    def get_ghostnet_se_ratio(cls):
        rs = cls.__read_ini_file('ghostnet_configuration', 'se_ratio')
        se_ratio = []
        for i in rs.split(','):
            se_ratio.append(float(i))
        return se_ratio

    @classmethod
    def get_init_params(cls):
        params = {}
        params['pop_size'] = cls.get_pop_size()
        params['max_gen'] = cls.get_max_generation()
        params['prob_crossover'] = cls.get_genetic_probability()[0]
        params['prob_mutation'] = cls.get_genetic_probability()[1]
        params['mutation_probs'] = cls.get_mutation_probs()

        params['stage_length_limit'] = cls.get_stage_length_limit()
        params['downsample_number'] = cls.get_downsample_number()
        params['image_channels'] = cls.get_input_channels()
        params['init_channels'] = cls.get_init_channels()
        params['image_size'] = cls.get_input_size()
        params['output_channels'] = cls.get_output_channels()
        # params['num_class'] = cls.get_num_class()

        params['mobilenet_expansion_factor'] = cls.get_mobilenet_expansion_factor()
        params['mobilenet_kernel_size'] = cls.get_mobilenet_kernel_size()
        params['mobilenet_se_ratio'] = cls.get_mobilenet_se_ratio()
        params['ghostnet_expansion_factor'] = cls.get_ghostnet_expansion_factor()
        params['ghostnet_kernel_size'] = cls.get_ghostnet_kernel_size()
        params['ghostnet_se_ratio'] = cls.get_ghostnet_se_ratio()
        return params


class Log(object):
    _logger = None

    @classmethod
    def __get_logger(cls):
        if Log._logger is None:
            logger = logging.getLogger("EZBNAS")
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
            file_handler = logging.FileHandler("main.log")
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            Log._logger = logger
            return logger
        else:
            return Log._logger

    @classmethod
    def info(cls, _str):  # normal info message
        cls.__get_logger().info('\t' + _str)

    @classmethod
    def warn(cls, _str):
        cls.__get_logger().warning('\t' + _str)

    @classmethod
    def important(cls, _str):  # important info message
        cls.__get_logger().info(_str)


class Utils(object):
    @classmethod
    def individuals_to_string(cls, individuals):
        _str = []
        for ind in individuals:
            _str.append(str(ind))
            _str.append('-' * 100)
        return '\n'.join(_str)

    @classmethod
    def save_individuals_at_begin(cls, individuals, gen_no):
        file_name = './populations/begin_%02d.txt' % (gen_no)
        with open(file_name, 'w') as f:
            f.write(cls.individuals_to_string(individuals))
            f.flush()

    @classmethod
    def save_individuals_after_crossover(cls, individuals, gen_no):
        file_name = './populations/crossover_%02d.txt' % (gen_no)
        with open(file_name, 'w') as f:
            f.write(cls.individuals_to_string(individuals))
            f.flush()

    @classmethod
    def save_individuals_after_mutation(cls, individuals, gen_no):
        file_name = './populations/mutation_%02d.txt' % (gen_no)
        with open(file_name, 'w') as f:
            f.write(cls.individuals_to_string(individuals))
            f.flush()

    @classmethod
    def save_fitness_after_evaluate(cls, individuals, gen_no):
        file_name = './populations/evaluate_%02d.txt' % (gen_no)
        with open(file_name, 'a') as f:
            for indi in individuals:
                f.write('%s:acc=%.5f;params=%.5f;\n' % (indi.id, indi.acc, indi.params))
            f.flush()

    @classmethod
    def save_score(cls, individuals, gen_no):
        file_name = './populations/score_%02d.txt' % (gen_no)
        with open(file_name, 'a') as f:
            for indi in individuals:
                f.write('%s:zen_score:%.5f;zico_score=%.5f;params=%.5f;flops=%.5f;\n' % (indi.id, indi.zen_score, indi.zico_score, indi.params, indi.flops))
            f.flush()

    @classmethod
    def save_offspring(cls, individuals, gen_no):
        file_name = './populations/offspring_%02d.txt' % (gen_no)
        with open(file_name, 'w') as f:
            f.write(cls.individuals_to_string(individuals))
            f.flush()

    @classmethod
    def save_best_individual(cls, individual, gen_no):
        file_name = './populations/best_%02d.txt' % (gen_no)
        with open(file_name, 'a') as f:
            f.write(cls.individuals_to_string([individual]))
            f.flush()

    @classmethod
    def get_newest_file_based_on_prefix(cls, prefix):
        id_list = []
        for _, _, file_names in os.walk('./populations'):
            for file_name in file_names:
                if file_name.startswith(prefix):
                    id_list.append(int(file_name[len(prefix) + 1:len(prefix) + 3]))
        if len(id_list) == 0:
            return None
        else:
            return np.max(id_list)

    @classmethod
    def load_population(cls, prefix, gen_no):
        file_name = './populations/%s_%02d.txt' % (prefix, gen_no)
        params = GlobalConfigTool.get_init_params()
        pop = Population(params, gen_no)

        # .txt key to init function key
        key_mapping = {
            "number": "_number",
            "in": "_in_channels",
            "out": "_out_channels",
            "exp": "_expansion_factor",
            "ks": "_kernel_size",
            "se": "_se_ratio",
            "type": "_downsample_type"
        }
        individual = None
        stage = None
        with open(file_name, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('--'):
                    stage.out_channels = stage.units[-1].out_channels
                    individual.stages.append(stage)
                    pop.individuals.append(individual)  # add one individual to population
                    pop.number_id += 1
                elif line.startswith('indi:'):
                    indi_no = line.split(':')[1]
                    stage = None
                    individual = Individual(params, indi_no)
                elif line.startswith('Acc:'):
                    individual.acc = float(line.split(':')[1])
                elif line.startswith('Params:'):
                    individual.params = float(line.split(':')[1])
                elif line.startswith('Flops:'):
                    individual.flops = float(line.split(':')[1])
                elif line.startswith('Zen:'):
                    individual.zen_score = float(line.split(':')[1])
                elif line.startswith('Zico:'):
                    individual.zico_score = float(line.split(':')[1])
                elif line.startswith('Params_Std:'):
                    pass
                elif line.startswith('{Stage'):
                    if stage is not None:
                        individual.stages.append(stage)
                    stage = Stage(individual.stage_id, params, params['output_channels'][individual.stage_id])
                    individual.stage_id += 1
                elif line.startswith('['):
                    unit_info = line.strip('[]}').split(',')
                    unit_type = unit_info[0]

                    # parse kwargs
                    kwargs = {}
                    for item in unit_info[1:]:
                        origin_key, origin_value = item.split(':')[0], item.split(':')[1]
                        key = key_mapping.get(origin_key, origin_key)
                        value = float(origin_value) if '.' in origin_value else int(origin_value)
                        kwargs[key] = value

                    if kwargs['_number'] == 0:
                        stage.in_channels = kwargs['_in_channels']

                    if unit_type == 'mobilenet':
                        unit = stage.init_a_mobile_unit(**kwargs)
                    elif unit_type == 'ghostnet':
                        unit = stage.init_a_ghost_unit(**kwargs)
                    else:
                        raise ValueError("Invalid unit type %s" % (unit_type))
                    stage.units.append(unit)  # add one unit to stage
                    stage.number_id += 1
                else:
                    raise ValueError("Invalid file content %s" % (line))
        return pop

    @classmethod
    def read_log_fitness(cls, indi_no):
        file_name = './log/%s.txt' % (indi_no)
        if not os.path.exists(file_name):
            return -1.0, -1.0, -1.0, -1.0, -1.0
        acc, params, flops, zen_score, zico_score = -1.0, -1.0, -1.0, -1.0, -1.0
        with open(file_name, 'r') as f:
            lines = f.readlines()[-5:]
            for line in lines:
                if "Finished-Acc" in line:
                    acc = float(line.split(':')[-1])
                if "Params" in line:
                    params = float(line.split(':')[-1])
                if "Flops" in line:
                    flops = float(line.split(':')[-1])
                if "Zen_score" in line:
                    zen_score = float(line.split(':')[-1])
                if "Zico_score" in line:
                    zico_score = float(line.split(':')[-1])


        return acc, params, flops, zen_score, zico_score

    @classmethod
    def read_template(cls):
        _path = './template/cifar.py'
        part1 = []
        part2 = []
        part3 = []
        part4 = []

        with open(_path, "r", encoding="utf-8") as f:
            f.readline()  # skip this comment
            line = f.readline().rstrip()
            while line.strip() != '# generated_init':
                part1.append(line)
                line = f.readline().rstrip()
            # print('\n'.join(part1))

            line = f.readline().rstrip()  # skip the comment '#generated_init'
            while line.strip() != '# generate_forward':
                part2.append(line)
                line = f.readline().rstrip()
            # print('\n'.join(part2))

            line = f.readline().rstrip()  # skip the comment '#generate_forward'
            while line.strip() != '# generate_forward_pre_gap':
                part3.append(line)
                line = f.readline().rstrip()

            line = f.readline().rstrip()  # skip the comment '# generate_forward_pre_gap'
            while line.strip() != '"""':
                part4.append(line)
                line = f.readline().rstrip()
            # print('\n'.join(part3))
        return part1, part2, part3, part4

    @classmethod
    def generate_pytorch_file(cls, indi, is_drop_path, no_reslink):
        # generate init list
        init_list = []
        # basic block
        image_channels = GlobalConfigTool.get_input_channels()
        init_channels = GlobalConfigTool.get_init_channels()

        init_list.append("self.basic_features = nn.Sequential(nn.Conv2d(%d, %d, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(%d), nn.ReLU(inplace=True))" % (
            image_channels, init_channels, init_channels))

        drop_path_rate = "self.drop_path_rate" if is_drop_path else "0.0"
        # evolved block
        for i in range(len(indi.stages)):
            init_list.append("# stage %d" % (i))
            stage = indi.stages[i]
            stage_length = len(stage.units)

            act_func = ['relu', 'relu', 'relu']

            for j in range(stage_length):
                stride = 2 if i != 0 and j == 0 else 1
                unit = stage.units[j]
                if unit.type == 1:
                    layer = "self.stage%d_block%d = MobileNetBottleneck(in_channels=%d, out_channels=%d, kernel_size=%d, expansion_factor=%d, stride=%d, se_ratio=%f, act_func='%s', no_reslink=%s)" % (
                        i, unit.number, unit.in_channels, unit.out_channels, unit.kernel_size, unit.expansion_factor, stride, unit.se_ratio, act_func[i], no_reslink)
                else:
                    layer = "self.stage%d_block%d = GhostBottleneck(in_channels=%d, out_channels=%d, kernel_size=%d, expansion_factor=%d, stride=%d, se_ratio=%f, act_func='%s', no_reslink=%s)" % (
                        i, unit.number, unit.in_channels, unit.out_channels, unit.kernel_size, unit.expansion_factor, stride, unit.se_ratio, act_func[i], no_reslink)
                init_list.append(layer)

        # final features
        end_channels = max(GlobalConfigTool.get_output_channels()[-1])
        init_list.append("# end")
        # init_list.append("self.end_features = nn.Sequential(nn.Conv2d(%d, %d, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(%d))" % (
        #     end_channels, end_channels, end_channels))
        init_list.append("self.end_features = nn.Sequential(nn.Conv2d(%d, %d, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(%d), nn.ReLU(inplace=True))" % (
            end_channels, end_channels, end_channels))

        # avg
        init_list.append("self.avg = nn.AdaptiveAvgPool2d(1)")

        # linear
        if GlobalConfigTool.get_dataset() == "cifar10":
            num_class = 10
        elif GlobalConfigTool.get_dataset() == "cifar100":
            num_class = 100
        else:
            num_class = 0
        init_list.append("self.fc = nn.Linear(%d, %d)" % (end_channels, num_class))

        # generate the forward part
        forward_list = []
        forward_list.append("out = self.basic_features(x)")
        # evolved block
        for i in range(len(indi.stages)):
            forward_list.append("# stage %d" % (i))
            for unit in indi.stages[i].units:
                forward_list.append("out = self.stage%d_block%d(out, %s)" % (i, unit.number, drop_path_rate))

        forward_list.append("# end")
        forward_list.append("out = self.end_features(out)")
        # forward_list.append("out = self.Hswish(out)")

        part1, part2, part3, part4 = cls.read_template()
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)

        for s in init_list:
            _str.append('        %s' % (s))

        _str.extend(part2)

        for s in forward_list:
            _str.append('        %s' % (s))

        _str.extend(part3)

        for s in forward_list:
            _str.append('        %s' % (s))

        _str.extend(part4)
        # print('\n'.join(_str))
        file_name = './scripts/%s.py' % (indi.id)
        script_file_handler = open(file_name, 'w', encoding="utf-8")
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()

    @classmethod
    def write_to_file(cls, _str, _file):
        with open(_file, 'w') as f:
            f.write(_str)
            f.flush()


if __name__ == '__main__':
    print(GlobalConfigTool.get_init_params())
    # print(Utils.read_template())
    ind = Individual(GlobalConfigTool.get_init_params(), 0)
    ind.initialize()
    print(ind)
    Utils.generate_pytorch_file(ind)
