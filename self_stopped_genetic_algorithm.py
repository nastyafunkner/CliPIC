import os
import numpy
import matplotlib.pyplot as plt
from os.path import join

from matplotlib import animation
from scipy.spatial.distance import euclidean

from _cluster_file_functions import extract_clusters
from _fit_functions import fit_alignment, fit_length
from _genetic_functions import generate_initial_population, crossover, mutation
from template_generating import pareto_frontier


def distance_to_zero(points_set):
    return sum([euclidean(p, (0, 0)) for p in points_set])


def pareto_distance(pareto1, pareto2):
    return abs(distance_to_zero(tuple(zip(*pareto1))) - distance_to_zero(tuple(zip(*pareto2))))


def selfstop_genetic_algorithm(word_clusters: list, project_path: str, alphabet: str,
                               increment=1.5,
                               # eps=0.1,
                               save_front_coefficient = 10,
                               mutation_probability=None, mutation_num=2,
                               start_population_factor=1.5,
                               parent_fraction=0.2,
                               size_low_threshold=5,
                               print_execution=True, print_gen_num=5,
                               animation_and_plot=True):
    """
    Реализован генетический алгоритм (ГА) для поиска оптимальных шаблонов
    :param save_front_coefficient: количество поколений,
    в течение которых фронт Парето остаётся неизменным, для остановки алгоритма
    :param eps: разница в растоянии двух последовательных фронтов
    Парето для остановки алгоритма
    :param word_clusters: кластер посл-тей
    :param project_path: путь к каталогу проекта
    :param alphabet: всевозможные состояния
    :param increment: во сколько раз длина шаблона может 
    превосходит самую длинную пос-ть в популяции
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

        population = list(set(word_cluster))
        population = population + generate_initial_population(size_start_population - len(population), alphabet,
                                                                mutation_probability, min_len,
                                                                max_len)


        prev_dist_to_zero = None
        save_pareto_count = 0
        generation = 0
        # pareto_diff = None
        # prev_parents_values_x = None
        # prev_parents_values_y = None
        # while pareto_diff is None or pareto_diff > eps:
        while save_pareto_count < save_front_coefficient:
            if print_execution:
                if (generation + 1) % print_gen_num == 0:
                    print(generation, 'generations done')
                else:
                    if prev_dist_to_zero is not None:
                        print(generation, '({})'.format(round(prev_dist_to_zero, 4)), end=' ... ')

            x_values = list([fit_alignment(word_cluster, p)[0] for p in population])
            y_values = list([fit_length(p, max_len) for p in population])

            points_position_for_animation.append(list(zip(x_values, y_values)))
            parents_values_x, parents_values_y, parents = pareto_frontier(x_values, y_values,
                                                                          population, False, False)

            # if pareto_diff is None and prev_parents_values_x is None:
            #     prev_parents_values_x = parents_values_x[:]
            #     prev_parents_values_y = parents_values_y[:]
            # else:
            #     pareto_diff = pareto_distance((parents_values_x, parents_values_y),
            #                                   (prev_parents_values_x, prev_parents_values_y))
            #     prev_parents_values_x, prev_parents_values_y = parents_values_x, parents_values_y

            if prev_dist_to_zero is None:
                prev_dist_to_zero = distance_to_zero(tuple(zip(*(parents_values_x, parents_values_y))))
                save_pareto_count = 1
            else:
                new_dist_to_zero = distance_to_zero(tuple(zip(*(parents_values_x, parents_values_y))))
                if new_dist_to_zero == prev_dist_to_zero:
                    save_pareto_count += 1
                else:
                    prev_dist_to_zero = new_dist_to_zero
                    save_pareto_count = 1

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
            generation += 1

        if print_execution:
            print()

        if animation_and_plot:
            fig = plt.figure(figsize=(5, 5))
            ax = plt.axes(xlim=(-0.1, 1.1), ylim=(-0.1, 3))
            plt.xlabel('Fit function 1: no aligned sequences')
            plt.ylabel('Fit function 2: length of tamplate')
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
                plt.savefig(join(path_new_directory, 'Plots_Pareto', '{}.png'.format(i)))

            anim = animation.FuncAnimation(fig, update,
                                           frames=len(points_position_for_animation), interval=500)
            anim.save(join(path_new_directory, 'animation_{}.mp4'.format(name_new_dir)), fps=2,
                      extra_args=['-vcodec', 'libx264'])

        file_with_result = open(join(path_new_directory,
                                     'result_templates_{}.txt'.format(name_new_dir)), 'w', encoding='utf-8')
        for pair in result:
            print(*pair, sep='\t', file=file_with_result)
        file_with_result.close()


if __name__ == '__main__':
    clusters_source = 'experiment/set1.txt'
    project_dir = 'experiment/test_dir'
    sequences = [list(set(extract_clusters(clusters_source, eval_=False)[0][0]))]
    selfstop_genetic_algorithm(sequences,
                               project_dir, alphabet='AFNIED',
                               increment=3, animation_and_plot=True)
