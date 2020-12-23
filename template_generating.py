import os
import numpy
import matplotlib.pyplot as pl

from matplotlib import animation
from os.path import join
from _fit_functions import fit_alignment, fit_length
from _genetic_functions import generate_initial_population, crossover, mutation
from _cluster_file_functions import extract_clusters


def pareto_frontier(xs: list, ys: list, seqs: list, max_x=True, max_y=True):
    """
    Определяет фронт Парето для набора точек
    :param xs: значения по x
    :param ys: значения по y
    :param seqs: соответствующие пос-ти
    :param max_x: искать максимум по x?
    :param max_y: искать максимум по y?
    :return: значения по x, по y, соответствующие 
    пос-ти фронта Парето
    """
    my_list = sorted([[xs[i], ys[i], seqs[i]]
                      for i in range(len(xs))], reverse=max_x)
    p_front = [my_list[0]]
    for pair in my_list[1:]:
        if max_y:
            if pair[1] >= p_front[-1][1]:
                p_front.append(pair)
        else:
            if pair[1] <= p_front[-1][1]:
                p_front.append(pair)
    p_front_x = [pair[0] for pair in p_front]
    p_front_y = [pair[1] for pair in p_front]
    p_front_seq = [pair[2] for pair in p_front]
    return p_front_x, p_front_y, p_front_seq


def genetic_algorithm(word_clusters: list, project_path: str, alphabet: str,
                      increment=1.5, max_num_generations=70,
                      mutation_probability=None, mutation_num=2,
                      start_population_factor=1.5,
                      parent_fraction=0.2,
                      size_low_threshold=5,
                      print_execution=True, print_gen_num=5,
                      animation_and_plot=True):
    """
    Реализован генетический алгоритм (ГА) для поиска оптимальных шаблонов
    :param word_clusters: кластер посл-тей
    :param project_path: путь к каталогу проекта
    :param alphabet: всевозможные состояния
    :param increment: во сколько раз длина шаблона может 
    превосходит самую длинную пос-ть в популяции
    :param max_num_generations: кол-во поколений для работы ГА
    :param mutation_probability: вероятности мутаций
    :param mutation_num: кол-во мутаций на посл-ть
    :param size_low_threshold: нижняя граница числа пос-тей.
    При её достижении увеличивается значение start_population_factor
    :param parent_fraction: доля родителей в популяции
    :param start_population_factor: во сколько раз превосходит 
    размер популяции размера кластера
    :param animation_and_plot: создавать ли анимации и графики фронтов Парето?
    :param print_gen_num: - 
    :param print_execution: выводить ли информацию о работе ГА?
    """

    if mutation_probability is None:
        mutation_probability = [0.3, 0.3, 0.3]

    for num_of_cluster, word_cluster in enumerate(word_clusters):

        name_new_dir = 'Cluster_' + str(num_of_cluster)
        path_new_directory = join(project_path, name_new_dir)

        if not os.path.exists(path_new_directory):
            os.makedirs(path_new_directory)

        if not os.path.exists(path_new_directory + '\\Plots_Pareto'):
            os.makedirs(path_new_directory + '\\Plots_Pareto')

        points_position_for_animation = []
        parents_position = []
        result = []

        if print_execution:
            print(len(word_cluster),
                  'sequences in cluster #{}'.format(num_of_cluster))

        if len(word_cluster) <= size_low_threshold:
            size_start_population = 20
        else:
            size_start_population = int(len(word_cluster) * start_population_factor)
        number_of_parents = int(size_start_population * parent_fraction)

        min_len = min([len(word) for word in word_cluster])
        max_len = max([len(word) for word in word_cluster])
        max_len += int(max_len * increment)

        alphabet = list(set(alphabet))

        population = word_cluster + generate_initial_population(size_start_population - len(word_cluster), alphabet,
                                                                mutation_probability, min_len,
                                                                max_len)

        for generation in range(max_num_generations):
            if print_execution:
                if (generation + 1) % print_gen_num == 0:
                    print(generation, 'generations done')
                else:
                    print(generation, end=' ... ')

            x_values = list([fit_alignment(word_cluster, p)[0] for p in population])
            y_values = list([fit_length(p, max_len) for p in population])

            points_position_for_animation.append(list(zip(x_values, y_values)))
            parents_values_x, parents_values_y, parents = pareto_frontier(x_values, y_values,
                                                                          population, False, False)

            result = [(p, fit_alignment(word_cluster, p, absolute=True)) for p in parents]
            parents_position.append([(fit_alignment(word_cluster, p)[0],
                                      fit_length(p, max_len)) for p in parents])

            while len(parents) < number_of_parents:
                while True:
                    applicant = numpy.random.choice(population)
                    if applicant not in parents:
                        parents.append(applicant)
                        break

            new_population = parents[:]
            while len(new_population) < len(population):
                while True:
                    child = crossover(numpy.random.choice(parents),
                                      numpy.random.choice(parents))
                    for t in range(mutation_num):
                        child = mutation(child, mutation_probability, alphabet)
                    if child not in new_population:
                        new_population.append(child)
                        break
            population = new_population

        if print_execution:
            print()

        if animation_and_plot:
            fig = pl.figure(figsize=(5, 5))
            ax = pl.axes(xlim=(-0.1, 1.1), ylim=(-0.1, 3))
            pl.xlabel('Fit function 1: no aligned sequences')
            pl.ylabel('Fit function 2: length of tamplate')
            x, y = list(zip(*(points_position_for_animation[0]
                              + parents_position[0])))
            scat = ax.scatter(x, y, lw=0, s=40,
                              c=[10] * len(points_position_for_animation[0])
                                + [100] * len(parents_position[0]),
                              cmap='winter')

            def update(i):
                new_set = points_position_for_animation[i] + parents_position[i]
                scat.set_offsets(new_set)
                colors = [10] * len(points_position_for_animation[i]) + [100] * len(parents_position[i])
                scat.set_array(numpy.array(colors))
                pl.savefig(join(path_new_directory, 'Plots_Pareto', '{}.png'.format(i)))

            anim = animation.FuncAnimation(fig, update,
                                           frames=len(points_position_for_animation), interval=500)
            anim.save(join(path_new_directory, 'animation_{}.mp4'.format(name_new_dir)), fps=2,
                      extra_args=['-vcodec', 'libx264'])

        file_with_result = open(join(path_new_directory,
                                     'result_templates_{}.txt'.format(name_new_dir)), 'w', encoding='utf-8')
        for pair in result:
            print(*pair, sep='\t', file=file_with_result)
        file_with_result.close()


def choose_best_templates(dir_from: str, dir_out: str, cluster_file: str,
                          genetic_file_name: str,
                          print_report=True, sep='\t'):
    """
    Выбирает лучшие шаблоны и сохраняет в отдельный файл
    :param genetic_file_name: название файла с генетическими шаблонами
    :param dir_from: директория с каталогами шаблонов для каждого кластера
    :param dir_out: директория для сохранения нового файла
    :param cluster_file: путь к файлу с кластерами
    :param print_report: выводить ли отчёт о подобранных шаблонах?
    :param sep: разделитель
    """
    clusters, names = extract_clusters(cluster_file, eval_=False)
    opened_file_out = open(join(dir_out, genetic_file_name), 'w', encoding='utf-8')
    if print_report:
        print('Cluster', 'Size', 'Aligned', sep='|')
        l1 = len('Cluster')
        l2 = len('Size')
        l3 = len('Aligned')
        f = '{:>' + str(l1) + '}' + '|' + '{:>' + str(l2) + '}' + '|' + '{:>' + str(l3) + '}'
    for i, cluster in enumerate(clusters):
        cluster_dir = 'Cluster_{}'.format(i)
        template, aligned = open(
            join(dir_from, cluster_dir, 'result_templates_{}.txt'.format(cluster_dir)), encoding='utf-8') \
            .readline().strip().split(sep)
        opened_file_out.write(template + '\n')
        if print_report:
            print(f.format(i, len(cluster), aligned))
    opened_file_out.close()


if __name__ == '__main__':
    clusters_source = 'data1/Clusters/patient_traces_test/KMeans8.txt'
    project_dir = 'data1/Template_generating'
    all_template_dir = 'data1'
    genetic_algorithm(extract_clusters(clusters_source, eval_=False)[0],
                      project_dir, alphabet='AFNIED', increment=3, animation_and_plot=False, max_num_generations=100)
    choose_best_templates(project_dir, all_template_dir, clusters_source)
