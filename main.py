import os
import clustering
import alignment
import fake_alignment
import cluster_visualization
import template_generating
import stat_calculator

from _cluster_file_functions import extract_clusters, record_clusters
from self_stopped_genetic_algorithm import selfstop_genetic_algorithm

SEQUENCE_SOURCE_DIR = 'data1/'
SEQUENCE_SOURCE_NAME = 'patient_traces_test'
OUT_DIR_CLUSTERS = 'data1/Clusters/'
OUT_DIR_VISUALIZATION = 'data1/Paths'
# OUTCOMES_FILE = 'data1/Outcomes.csv'
OUTCOMES_FILE = None
NODE_PARAMETERS_DIR = '_nodes_parameters/clinical_pathways_params'
EXPERT_TEMPLATE_FILE = 'data1/expert_templates.txt'
GENETIC_TEMPLATE_FILE = 'data1/genetic_templates.txt'
ALPHABET = 'AFNIED'
SEQUENCE_SEPARATOR = '\t'


if __name__ == '__main__':

    sequence_source_path = os.path.join(SEQUENCE_SOURCE_DIR, SEQUENCE_SOURCE_NAME + '.txt')
    OUT_DIR_CLUSTERS += SEQUENCE_SOURCE_NAME + '/'
    distance_matrix_file = '{0}{1}_similarity_matrix.txt'.format(OUT_DIR_CLUSTERS,
                                                                 SEQUENCE_SOURCE_NAME)

    vectors, cases = extract_clusters(sequence_source_path, eval_=False, header=True,
                                      encoding='cp1251', sep=SEQUENCE_SEPARATOR)
    vectors = vectors[0]
    cases = cases[0]
    print(len(vectors), 'chains')
    if len(vectors) > 25:

        lev_sim_matrix = clustering.get_similarity_matrix(distance_matrix_file, vectors)

        v, s = clustering.cluster_in_range(3, 20, lev_sim_matrix, vectors, cases, OUT_DIR_CLUSTERS)
        print('Look the plots of metrics and define the best number of clusters.')
        print('Close the plots before entering the number!')
        clustering.visualize_cluster_metrics(3, OUT_DIR_CLUSTERS, [v, s], ['variance', 'silhouette'])


        num_of_clusters = input('Enter the number of clusters: ')

        cluster_source = '{}KMeans{}.txt'.format(OUT_DIR_CLUSTERS, num_of_clusters)
        clusters, cluster_cases = extract_clusters(cluster_source, eval_=False, sep='\t')
    else:
        cluster_source = sequence_source_path
        clusters, cluster_cases = [vectors], [cases]

    choice_fake_alig = input('Use fake alignment? (Y/N)').lower() == 'y'
    choice_full_alig = input('Use full alignment? (Y/N)').lower() == 'y'

    if choice_fake_alig:
        file_alignment = cluster_source.split('.')[0] + '_fake_alignment.txt'
        aligned_clusters = [[] for i in range(len(clusters))]
        for i, cluster in enumerate(clusters):
            cyclic_alignment = fake_alignment.CyclicAlignment()
            for sequence in cluster:
                aligned_clusters[i].append(cyclic_alignment.align(sequence))

        record_clusters(file_alignment, aligned_clusters, cluster_cases, sep='\t')

        for i, cluster in enumerate(aligned_clusters):
            graph = cluster_visualization.Graph(cluster, cluster_cases[i],
                                                cluster_visualization.get_file_name(file_alignment), i, 1,
                                                NODE_PARAMETERS_DIR, OUTCOMES_FILE)
            graph.collect_edges()
            graph.to_gv(OUT_DIR_VISUALIZATION, 0.0, 0.3)

    if choice_full_alig:
        genetic_choice = input('Use experts or genetic templates? (E/G)').lower() == 'g'

        template_file = EXPERT_TEMPLATE_FILE
        if genetic_choice:

            template_dir = os.path.join(SEQUENCE_SOURCE_DIR, 'Template_generating', SEQUENCE_SOURCE_NAME,
                                        os.path.basename(cluster_source.split('.')[0]))
            selfstop_genetic_algorithm(clusters, template_dir,
                                       alphabet=ALPHABET, mutation_num=10,
                                       increment=10, animation_and_plot=False)
            template_generating.choose_best_templates(template_dir, SEQUENCE_SOURCE_DIR, cluster_source,
                                                      os.path.basename(GENETIC_TEMPLATE_FILE))

            template_file = GENETIC_TEMPLATE_FILE

        file_out = cluster_source.split('.')[0] + '_full_alignment.txt'
        num_state = alignment.align_all(clusters, cluster_cases, file_out, template_file)
        this_clusters, this_cases = extract_clusters(file_out, sep='\t', eval_=False)

        for i, cluster in enumerate(this_clusters):

            graph = cluster_visualization.Graph(cluster, this_cases[i],
                                                cluster_visualization.get_file_name(
                                                    file_out), i, num_state, NODE_PARAMETERS_DIR, OUTCOMES_FILE,
                                                add_figures=False)
            graph.collect_edges()
            try:
                graph.to_gv(OUT_DIR_VISUALIZATION, 0, 0.1)
            except AssertionError:
                print('Cluster #{} is empty'.format(i))

