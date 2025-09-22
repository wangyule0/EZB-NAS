import numpy as np

from utils import GlobalConfigTool


class Selection(object):
    def __init__(self):
        self.params_min, self.params_max = GlobalConfigTool.get_params_limit()

    def roulette_selection(self, _a, k):
        a = np.asarray(_a)
        idx = np.argsort(a)[::-1]  # descending order index array
        sort_a = a[idx]  # descending order data array  [5, 3, 2, 1]
        sum_a = float(np.sum(sort_a))   # sum
        selected_index = []

        cumulative_sum = np.cumsum(sort_a)  # get cumulative sum array  [5, 8, 10, 11]

        for _ in range(k):
            u = np.random.rand() * sum_a
            pos = np.searchsorted(cumulative_sum, u)    # search index by random sum
            if pos == len(idx):     # make sure pos valid   when all value of a < 0
                pos = len(idx) - 1
            selected_index.append(idx[pos])

        if k == 1:
            selected_index = selected_index[0]
        return selected_index

    def ramdom_selection(self, _a, k):
        selected_index = []
        for _ in range(k):
            selected_index.append(np.random.randint(0, len(_a)))
        if k == 1:
            selected_index = selected_index[0]
        return selected_index

    def max_selection(self, _a, k):
        selected_index = sorted(range(len(_a)), key=lambda i: _a[i], reverse=True)[:k]
        if k == 1:
            selected_index = selected_index[0]
        return selected_index

    def fast_non_dominated_sort(self, individuals, objectives, maximize):
        fronts = [[]]
        for p in individuals:
            p.domination_count = 0
            p.dominated_solutions = []

            for q in individuals:
                if self.dominates(p, q, objectives, maximize):
                    p.dominated_solutions.append(q)
                elif self.dominates(q, p, objectives, maximize):
                    p.domination_count += 1

            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]

    def dominates(self, p, q, objectives, maximize):
        not_worse = True
        strictly_better = False

        for obj in objectives:
            p_value = getattr(p, obj)
            q_value = getattr(q, obj)

            if obj in maximize:
                if p_value < q_value:
                    not_worse = False
                if p_value > q_value:
                    strictly_better = True
            else:
                if p_value > q_value:
                    not_worse = False
                if p_value < q_value:
                    strictly_better = True

        return not_worse and strictly_better

    # 拥挤度计算
    def crowding_distance_assignment(self, front, objectives):
        n = len(front)
        if n == 0:
            return

        for ind in front:
            ind.crowding_distance = 0

        for key in objectives:
            front.sort(key=lambda ind: getattr(ind, key))

            front[0].crowding_distance = front[-1].crowding_distance = float("inf")

            min_f, max_f = getattr(front[0], key), getattr(front[-1], key)

            if max_f - min_f == 0:
                continue

            for i in range(1, n - 1):
                prev_f = getattr(front[i - 1], key)
                next_f = getattr(front[i + 1], key)

                front[i].crowding_distance += (next_f - prev_f) / (max_f - min_f)

    def nsga2_sort(self, individuals, objectives, maximize):
        fronts = self.fast_non_dominated_sort(individuals, objectives, maximize)
        for front in fronts:
            self.crowding_distance_assignment(front, objectives)

    def nsga2_selection(self, individuals, k, objectives, maximize):
        fronts = self.fast_non_dominated_sort(individuals, objectives, maximize)
        selected = []
        i = 0
        while len(selected) + len(fronts[i]) <= k:
            for ind in fronts[i]:
                selected.append(ind)
            i += 1

        if len(selected) < k:
            self.crowding_distance_assignment(fronts[i], objectives)
            fronts[i].sort(key=lambda ind: ind.crowding_distance, reverse=True)
            selected.extend(fronts[i][: k - len(selected)])

        if k == 1:
            selected = selected[0]
        return selected

    def nsga2_best_individuals(self, individuals, objectives, maximize, best_objective):
        fronts = self.fast_non_dominated_sort(individuals, objectives, maximize)
        for front in fronts:
            front = sorted(front, key=lambda indi: getattr(indi, best_objective), reverse=True)
            for indi in front:
                if self.params_min <= indi.params <= self.params_max:
                    return indi
        return None

    def nsga2_compare(self, indi1, indi2):
        # 比较 rank，如果不相同，直接返回较小的 rank 对应的个体
        if indi1.rank != indi2.rank:
            return indi1 if indi1.rank < indi2.rank else indi2
            # 如果 rank 相同，比较 crowding_distance
        return indi1 if indi1.crowding_distance > indi2.crowding_distance else indi2
