import os
import sklearn.cluster as clst
import numpy as np
import matplotlib.pyplot as plt
import _cluster_number_metrics as cnm
import warnings

from sklearn.metrics import silhouette_score
from jellyfish import levenshtein_distance
from _cluster_file_functions import extract_clusters, record_clusters

warnings.filterwarnings('ignore')


def calculate_lev_matrix(vectors_):
    """
    Функция для вычисления матрицы расстояний с расстоянием Левинштайна
    :param vectors_: векторы
    :return: матрицу расстояний
    """
    result = [[0] * len(vectors_) for _ in range(len(vectors_))]
    for i in range(len(vectors_)):
        print(i, 'lines in matrix calculated')
        for j in range(i + 1, len(vectors_)):
            result[i][j] = result[j][i] = levenshtein_distance(vectors_[i], vectors_[j])
    return result


def cluster_in_range(low_num_clusters, high_num_cluster, vectors_,
                     init_vectors_, cases_, out_dir_, random_state=42):
    """
    Проводит кластеризацию k-means для числа кластеров
    от low_num_clusters до high_num_clusters.
    Кластера сохраняются в директорию out_dir.
    Каждый резултат кластеризации сохраняется в отдельном файле.
    :param cases_: имена объектов для кластеризации
    :param init_vectors_: список векторов, которые
    будут добавлены в файлы с кластерами
    :param random_state: можно выбрать
    :param low_num_clusters: нижняя граница числа кластеров
    :param high_num_cluster: верхняя граница числа кластеров
    :param vectors_: объекты для кластеризации
    :param out_dir_: директория для сохранения кластеров
    :return: 2 списка с метриками кластеризации variance и silhouette,
    которые могут быть дальше использованы для определения чила кластеров
    """
    assert low_num_clusters >= 3, \
        'Low bound of clusters number should be 3 or more'

    if not os.path.exists(out_dir_):
        os.makedirs(out_dir_)

    variance = []
    silhouette = []
    centers = []
    for k in range(low_num_clusters, high_num_cluster + 1):
        print(k, 'clusters now')
        model = clst.KMeans(n_clusters=k, random_state=random_state)
        labels = model.fit_predict(vectors_)

        shift = min(labels) * (-1)
        for i in range(len(labels)):
            labels[i] += shift

        clusters = [[] for i in range(max(labels) + 1)]
        clusters_of_case = [[] for i in range(max(labels) + 1)]
        clusters_from_matrix = [[] for i in range(max(labels) + 1)]
        for i in range(len(labels)):
            clusters[labels[i]].append(init_vectors_[i])
            clusters_of_case[labels[i]].append(cases_[i])
            clusters_from_matrix[labels[i]].append(vectors_[i])
        variance.append(cnm.beta_CV(clusters_from_matrix, model.cluster_centers_))
        centers.append(model.cluster_centers_)
        silhouette.append(silhouette_score(np.array(vectors_), labels))

        m = str(model)[:str(model).find('(')]
        path_file = out_dir_ + m + str(k) + '.txt'
        record_clusters(path_file, clusters, clusters_of_case, sep='\t')
    return variance, silhouette


def visualize_cluster_metrics(low_num_clusters,
                              out_dir_, metrics: list, labels: list, show=True):
    """
    Строит и сохряанет графики с метриками
    :param low_num_clusters: нижняя граница числа кластеров,
    которая использовалась при кластеризации
    :param out_dir_: директория для сохранения графика
    :param metrics: метрики для визуализации
    :param labels: название каждой метрики
    :param show: показать ли график во время работы программы
    или только сохранить
    :return:
    """
    colors = ['b', 'g', 'r']
    num_metrics = len(metrics)
    assert len(metrics) <= num_metrics, 'Too many metrics'
    plt.figure(figsize=(5 * num_metrics, 5))
    num_subplots = 100 + 10 * num_metrics + 1
    subplots = list(range(num_subplots, num_subplots + num_metrics))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        plt.subplot(subplots[i])
        x = range(low_num_clusters, low_num_clusters + len(metric))
        plt.plot(x, metric, colors[i], label=label)
        plt.plot(x, metric, colors[i] + 'o')
        plt.xlabel('Number of clusters')
        plt.legend()
        plt.xticks(x, x)
    plt.savefig(
        out_dir_ + 'Metrics_for_clusters.png')
    if show:
        plt.show()
    plt.clf()


def safe_in_file_metrics(out_dir_, metrics: list, labels: list):
    """
    Сохраняет значение метрик в отдельный файл
    :param out_dir_: директория для сохранения
    :param metrics: список метрик
    :param labels: название метрик
    :return:
    """
    metrics_file_open = open(out_dir_ + 'cluster_metrics.txt', 'w')
    for metric, label in zip(metrics, labels):
        print(label, metric, sep='\t', file=metrics_file_open)
        print(label, metric, sep='\t', file=metrics_file_open)
    metrics_file_open.close()


def get_similarity_matrix(distance_matrix_file_, vectors_):
    if os.path.exists(distance_matrix_file_):
        lev_sim_matrix_ = [[int(num) for num in line.split(';')]
                           for line in open(distance_matrix_file_)]
    else:
        if not os.path.exists(os.path.dirname(distance_matrix_file_)):
            os.makedirs(os.path.split(distance_matrix_file_)[0])
        lev_sim_matrix_ = np.array(calculate_lev_matrix(vectors_))
        for i in range(len(lev_sim_matrix_)):
            print(*lev_sim_matrix_[i], sep=';',
                  file=open(distance_matrix_file_, 'a'))
    return lev_sim_matrix_


if __name__ == '__main__':
    sequence_source_dir = 'data1/'
    sequence_source_name = 'patient_traces_test'
    sequence_source_path = sequence_source_dir + sequence_source_name + '.txt'
    out_dir = 'data1/Clusters/' + sequence_source_name + '/'
    distance_matrix_file = '{0}{1}_similarity_matrix.txt'.format(out_dir,
                                                                 sequence_source_name)

    vectors, cases = extract_clusters(sequence_source_path, eval_=False)
    vectors = vectors[0]
    cases = cases[0]

    lev_sim_matrix = get_similarity_matrix(distance_matrix_file, vectors)

    v, s = cluster_in_range(3, 8, lev_sim_matrix, vectors, cases, out_dir)
    visualize_cluster_metrics(3, out_dir, [v, s], ['variance', 'silhouette'])
    safe_in_file_metrics(out_dir, [v, s], ['variance', 'silhouette'])
