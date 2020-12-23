import pandas as pd


def extract_clusters(path_to_file: str, sep='\t', eval_=True, header=True, encoding='cp1251'):
    """
    Подается файл, в котором каждый кластер отделен словом Cluster
    Каждая строчка между кластерами содержит последовательность и номер
    объекта, которму она соответствует
    :param encoding: кодировка
    :param path_to_file: путь к файлу с кластерами
    :param sep: символ-разделитель между последовательностю и её именем
    :param eval_: выполняет строку кода с последовательностью (e.g. списком)
    :param header: является ли первая строка файла загаловком
    :return: два списка кластеров последовательностей и их имён
    """
    clusters = []
    cluster = []
    all_cases = []
    cases = []

    for line in open(path_to_file, encoding=encoding):
        line = line.strip()
        if header:
            header = False
            continue
        if 'Cluster' in line:
            clusters.append(cluster)
            cluster = []
            all_cases.append(cases)
            cases = []
        else:
            sequence = line.split(sep)[0]
            if eval_:
                cluster.append(eval(sequence))
            else:
                cluster.append(sequence)
            cases.append(line.split(sep)[1])
    clusters.append(cluster)
    all_cases.append(cases)
    return clusters, all_cases


def record_clusters(path_out: str, clusters_out: list, cases_out: list, sep: str, spec_title=None, encoding='cp1251'):
    """
    Записывает кластера в файл
    :param path_out: путь к файлу записи
    :param clusters_out: кластера объектов для записи
    :param cases_out: кластера имен объектов для записи
    :param sep: разделитель между объектами и их именами
    :param spec_title: пользовательские заголовки кластеров
    :return:
    """
    file_out_open = open(path_out, 'w', encoding=encoding)

    for i, cluster in enumerate(clusters_out):
        if spec_title:
            print(spec_title[i], file=file_out_open)
        else:
            print('Cluster', i, len(cluster), file=file_out_open)
        for line in zip(cluster, cases_out[i]):
            print(*line, sep=sep, file=file_out_open)

    file_out_open.close()


def join_lists(input_lists):
    new_list = []
    for this_list in input_lists:
        new_list += this_list
    return new_list


def clusters_to_pd(cluster_file):
    """
    Преобразует файл с кластерами TXT в pandas.Dataframe
    :param cluster_file: файл с кластерами
    :return: pd.Dataframe
    """
    clusters, cases = extract_clusters(cluster_file, eval_=False)
    num_clusters = [[i] * len(cases) for i, cases in enumerate(cases)]
    num_clusters = join_lists(num_clusters)
    cases = join_lists(cases)
    clusters = join_lists(clusters)
    clusters_df = pd.DataFrame(columns=['id',
                                        'chain',
                                        'cluster',
                                        ])
    clusters_df['id'] = cases
    clusters_df['chain'] = clusters
    clusters_df['cluster'] = num_clusters
    return clusters_df


def pd_to_clusters(clusters_df):
    """
    Преобразует pandas.Dataframe ('id', 'chain', 'cluster') в файл с кластерами TXT
    :return: clusters, cases
    """
    clusters = [[] for _ in range(clusters_df.cluster.max() + 1)]
    cases = [[] for _ in range(len(clusters))]
    print(clusters, cases)
    for _, row in clusters_df.iterrows():
        clusters[int(row.cluster)].append(row.chain)
        cases[int(row.cluster)].append(row.id)
    return clusters, cases
