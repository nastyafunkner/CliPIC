import os
import matplotlib.cm as cm
import numpy as np
import graphviz

from shutil import copyfile

from _cluster_file_functions import extract_clusters


class Graph:
    """
    Класс для визуализации клиичсеких путей одного кластера
    """

    def __init__(self, cluster_: list, cases_: list, file_id: str, cluster_id,
                 num_digits: int, nodes_param_path: str,
                 outcome_file=None, add_figures=False):
        """
        :type add_figures: добавлять ли в вершины картинки
        :type file_id: идентификатор происхождения кластера
        (может быть имя файла кластеров)
        :param cluster_: кластер объектов
        :param cases_: клатер имен объектов
        :param cluster_id: идентификатор кластера
        :param num_digits: число цифр отведенных под номер состояния
        :param outcome_file: файл с исходами пациентов, может отсутствовать
        :param nodes_param_path: путь к директории с фалами настройки для отрисовки узлов графов:
        colors_nodes.txt, names_dict.txt, outcomes_ru_eng(доп. - может отсутствовать)
        """

        self.cluster = cluster_
        self.cases = cases_
        self.num_digits = num_digits
        self.start_symbol = '_' + '0' * num_digits
        self.end_symbol = '*' + '0' * num_digits
        self.size_node = self.num_digits + 1
        self.edges = {}

        self.node_colors, self.node_names, self.outcomes_ru_eng = self.__set_node_params(nodes_param_path)
        self.add_figures = add_figures
        self.figure_dir = os.path.join(nodes_param_path, 'node_figures')

        self.outcome_file = outcome_file
        self.outcomes = {}
        if self.outcome_file:
            self.__get_outcomes()
        self.cluster_id = cluster_id
        self.file_id = file_id

    def collect_edges(self):
        """
        Формирует граф по предоставленным объектам кластера
        :return:
        """
        for sequence in self.cluster:
            sequence = self.start_symbol + sequence + self.end_symbol
            for i in range(0, len(sequence) - self.size_node, self.size_node):
                pair = sequence[i:i + self.size_node * 2]
                if pair not in self.edges:
                    self.edges[pair] = 0
                self.edges[pair] += 1

    def to_gv(self, out_dir_, threshold, significance):
        """
        Формирует файл .gv с графом
        Для просмотра файлов gv необходим Graphviz
        :param out_dir_: директория для сохранения графа
        :param threshold: будут добавлены только ребра,
        вес которых >= threshold * len(cluster)
        :param significance: ребра, вес которых >= significance * len(cluster),
        будут иметь большую толщину
        :return: name of the gv file
        """
        assert len(self.edges) > 0, 'At first, collect edges'

        if not os.path.exists(out_dir_):
            os.makedirs(out_dir_)
        additional_dir = '{}/{}/Threshold_{}'.format(out_dir_, self.file_id, threshold)
        if not os.path.exists(additional_dir):
            os.makedirs(additional_dir)
        if self.add_figures:
            for file in os.listdir(self.figure_dir):
                copyfile(os.path.join(self.figure_dir, file), os.path.join(additional_dir, file))
        
        gv_name = '{}/Cluster_{}.gv'.format(additional_dir, self.cluster_id)
        opened_gv = open(gv_name, 'w')
        added_nodes = self.__begin_gv(opened_gv)
        self.__add_edges_to_gv(opened_gv, added_nodes, len(self.cluster) * threshold,
                               round(len(self.cluster) * significance))
        self.__add_outcomes(opened_gv)
        self.__add_name_node(opened_gv)
        self.__finish_gv(opened_gv)
        self.__render_gv(gv_name)
        
        return gv_name

    def __finish_gv(self, opened_gv):
        opened_gv.write('}')
        opened_gv.close()

    def __add_name_node(self, opened_gv):
        self.__add_node_to_gv(opened_gv, 'Cluster {} ({})'.format(self.cluster_id, len(self.cases)),
                              [['color', 'white'], ['fontsize', 20],
#                                ['fontname', "Times New Roman"],
                              ], standard=False)

    def __add_outcomes(self, opened_gv):
        if self.outcomes:
            outcome_line = self.__get_outcomes_line()
            self.__add_node_to_gv(opened_gv, 'result',
                                  [['label', outcome_line],
                                   ['shape', 'rectangle'],
#                                    ['fontname', "Times New Roman"],
                                  ],
                                  standard=False)
            opened_gv.write('"{}"->"result";'.format(self.end_symbol))

    def __begin_gv(self, opened_gv):
        opened_gv.write('digraph G{' + '\n')
        opened_gv.write('rankdir=LR;' + '\n')
        self.__add_node_to_gv(opened_gv, self.start_symbol, [
            ['label', '{}\n{}'.format(self.node_names[self.start_symbol[0]],
                                      len(self.cluster))],
            ['fillcolor', self.node_colors[self.start_symbol[0]]],
#             ['fontname', "Times New Roman"],
            ['style', 'filled'],
            ['shape', 'rectangle']])
        self.__add_node_to_gv(opened_gv, self.end_symbol, [])
        added_nodes = [self.start_symbol, self.end_symbol]
        return added_nodes

    def __add_node_to_gv(self, opened_gv, node, attributes, standard=True):
        if standard:
            attributes += self.__standard_attributes(node)
        opened_gv.write('"{}"{}\n'
                        .format(node, self.__form_attribute(attributes)))

    def __standard_attributes(self, node):
        if self.add_figures:
            return [
                ['label', ''],
                # ['fillcolor', self.node_colors[node[0]]],
#                 ['fontname', "Times New Roman"],
                ['style', 'filled'],
                ['shape', 'rectangle'],
                ['image', '{}.png'.format(self.node_names[node[0]])]
            ]
        return [
            ['label', self.node_names[node[0]]],
            ['fillcolor', self.node_colors[node[0]]],
#             ['fontname', "Times New Roman"],
            ['style', 'filled'],
            ['shape', 'rectangle']
        ]

    # def __get_outcomes(self):
    #     for line in open(self.outcome_file):
    #         case = line.split(';')[0]
    #         outcome = line.split(';')[5]
    #         where = line.split(';')[6]
    #         if 'Выписан' in outcome:
    #             if 'лечение' in where or not where:
    #                 self.outcomes[case] = 'на лечение'
    #             else:
    #                 self.outcomes[case] = where
    #         elif not outcome or 'Улучшение' in outcome:
    #             self.outcomes[case] = 'на лечение'
    #         else:
    #             self.outcomes[case] = outcome.lower()
    #
    #
    #     if self.outcomes_ru_eng is not None:
    #         for key in self.outcomes:
    #             self.outcomes[key] = self.outcomes_ru_eng[self.outcomes[key]]
    #
    #     return self.outcomes

    def __get_outcomes_line(self):
        out = {}
        for case in self.cases:
            if case not in self.outcomes:
                continue
            if self.outcomes[case] not in out:
                out[self.outcomes[case]] = 0
            out[self.outcomes[case]] += 1
        return '\n'.join(['{}:{}'.format(key, out[key]) for key in out])

    def __get_outcomes(self):
        data, names = extract_clusters(self.outcome_file, sep='\t', eval_=False)
        data = data[0]
        names = names[0]
        self.outcomes = dict(zip(names, data))
        return self.outcomes

    # def __get_outcomes_line(self):
    #     out = {}
    #     for case in self.cases:
    #         if case not in self.outcomes:
    #             continue
    #         categories = self.outcomes[case] \
    #             .replace('М', 'M') \
    #             .replace('Е', 'E') \
    #             .replace('К', 'K') \
    #             .split('|')
    #         for c in categories:
    #             if c not in out:
    #                 out[c] = 0
    #             out[c] += 1
    #     sorted_keys = sorted(out.items(), key=lambda x: x[1], reverse=True)
    #     sorted_keys = [s[0] for s in sorted_keys]
    #     return '\n'.join(['{}:{}'.format(key, out[key]) for key in sorted_keys])

    def __add_edges_to_gv(self, file, added_nodes, threshold, significance):
        for edge in self.edges:
            if self.edges[edge] >= threshold:
                state1 = edge[:len(edge) // 2]
                state2 = edge[len(edge) // 2:]
                if state1 not in added_nodes:
                    self.__add_node_to_gv(file, state1, [])
                    added_nodes.append(state1)
                if state2 not in added_nodes:
                    self.__add_node_to_gv(file, state2, [])
                    added_nodes.append(state2)
                label = str(int(round(self.edges[edge], 2)))
                penwidth = 1
                fontsize = 20
                if self.edges[edge] >= significance:
                    penwidth = 5
                    fontsize = 30
                attributes = [['label', label],
                              ['fontsize', fontsize],
                              ['penwidth', penwidth]]
                file.write('"{}"->"{}"{}\n'.format(state1, state2,
                                                   self.__form_attribute(attributes)))

    def __form_attribute(self, attributes):
        if attributes:
            attr = ['{}="{}"'.format(a[0], a[1]) for a in attributes]
            return '[{}];'.format(','.join(attr))
        return ''

    def __set_node_params(self, nodes_param_path):
        node_names = eval(open(os.path.join(nodes_param_path, 'names_dict.txt'), encoding='utf8').read())
        try:
            node_colors = eval(open(os.path.join(nodes_param_path, 'colors_nodes.txt')).read())
        except FileNotFoundError:
            if len(node_names) <= 25:
                bytes_colors = cm.rainbow(np.linspace(0, 1, len(node_names) - 2), bytes=True)
                hex_colors = iter(['#%02x%02x%02x' % tuple(b[:3]) for b in bytes_colors])
                node_colors = {key: next(hex_colors) for key in node_names.keys() if key not in ('_', '*')}
                node_colors.update({'_': 'white',
                                    '*': 'white', })
            else:
                node_colors = {key: 'white' for key in node_names}
        if 'outcomes_ru_eng.txt' in os.listdir(nodes_param_path):
            outcomes_ru_eng = eval(open(os.path.join(nodes_param_path, 'outcomes_ru_eng.txt')).read())
        else:
            outcomes_ru_eng = None
        return node_colors, node_names, outcomes_ru_eng

    def __render_gv(self, gv_name):
        graphviz.render('dot', 'png', gv_name)


def get_file_name(path):
    return path[path.rfind('/') + 1:path.rfind('.')]


if __name__ == '__main__':
    cluster_file = 'data1/Clusters/patient_traces_test/KMeans8_fake_alignment.txt'
    digit_nums = 1  # Обычно для fake = 1, для full = 2
    out_dir = 'data1/Paths'
    outcomes_file = 'data1/Outcomes.csv'
    node_param_dict = '_nodes_parameters/clinical_pathways_params'

    clusters, cases = extract_clusters(cluster_file, sep='\t', eval_=False)

    for i, cluster in enumerate(clusters):
        graph = Graph(cluster, cases[i],
                      get_file_name(cluster_file), i, digit_nums, node_param_dict, outcomes_file)
        graph.collect_edges()
        graph.to_gv(out_dir, 0.0, 0.3)
