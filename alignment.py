import math

from _cluster_file_functions import extract_clusters, record_clusters


def align_word(word: str, template: str, print_execution=True):
    """
    Выравнивает строку word по шаблону template
    :param word: последователньость состояний
    :param template: шаблон для выравнивания
    :return: None, если не возможно выравнить,
    иначе выравненную строку
    :param print_execution:
    """
    align = ''
    location = 0
    for letter in word:
        for loc in range(location, len(template)):
            if letter == template[loc]:
                align += letter
                location = loc + 1
                break
            else:
                align += '_'
        else:
            if print_execution:
                print('NON-ALIGNED:', word, align)
            return None
    if 0 < len(align) < len(template):
        align += '_' * (len(template) - len(align))
    return align


def add_state_frequency(sequence: str, n: int):
    """
    Преобразует выравненную строку к строке
    с пронумерованными состояними по шаблону
    :param sequence: выравненная строка
    :param n: число символов под номер состояния
    :return: преобразованную строку
    """
    new_sequence = ''
    template = "%0" + str(n) + "d"
    for i, symbol in enumerate(sequence):
        if symbol != '_':
            new_sequence += symbol + template % i
    return new_sequence


def define_templates(template_file_: str):
    """
    Определяет шаблоны для выравнивания
    :param template_file_: путь к файлу с шаблонами
    :return: список шаблонов для каждого кластера
    """
    expert_templates = list(map(lambda s: s.strip(),
                                open(template_file_, encoding='utf-8').readlines()))
    return expert_templates


def define_number_for_state(templates: list):
    """
    Определяет необходимое число символов для нумерации
    состояний в выравненных последовательностях
    :param templates: список шаблонов
    :return:
    """
    max_len = max([len(t) for t in templates])
    return int(math.log(max_len, 10)) + 1


def align_all(clusters_: list, cases_: list,
              file_out_: str, template_file_: str):
    """
    Выравнивает все последовательности всех кластеров по шаблонам
    :param clusters_: список объектов по кластерам
    :param cases_: список имен объектов по кластерам
    :param file_out_: путь к файлы для записи выравненных кластеров
    :param template_file_: путь к файлу с шаблонами
    :return:
    """
    templates = define_templates(template_file_)
    num_state = define_number_for_state(templates)
    align_clusters = []
    align_cases = []
    titles = []

    for i, cluster in enumerate(clusters_):
        align_cluster = []
        align_cases_this = []
        if not templates[i]:
            align_clusters.append([])
            continue
        for j, word in enumerate(cluster):
            align_result = align_word(word, templates[i])
            if align_result:
                align_cluster.append(add_state_frequency(align_result,
                                                         num_state))
                align_cases_this.append(cases_[i][j])
        titles.append('Cluster {0} ({1}) Template: {2}'
                      .format(i, len(align_cluster), templates[i]))
        align_clusters.append(align_cluster)
        align_cases.append(align_cases_this)

    record_clusters(file_out_, align_clusters,
                    align_cases, sep='\t', spec_title=titles)
    return num_state


if __name__ == '__main__':
    template_file = 'data1/expert_templates.txt'
    clusters_file = 'data1/Clusters/patient_traces_test/KMeans8.txt'
    file_out = clusters_file.split('.')[0] + '_full_alignment.txt'

    clusters, cases = extract_clusters(clusters_file,
                                       sep='\t', eval_=False)
    align_all(clusters, cases, file_out, template_file)
