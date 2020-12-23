from scipy.spatial.distance import euclidean as euclid_distance

"""
Реализация формул для определения числа кластеров из статьи:
Determination of Number of Clusters in K-Means Clustering and
Application in Colour Image Segmentation
http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.587.3517
"""


def average_intercluster_distance(cluster, center):
    """
    dk with a wave
    """
    d_k = sum([euclid_distance(x, center) for x in cluster]) / len(cluster)
    return d_k


def intercluster_distance_between_cluster(center1, center2):
    """
    Dij with a wave
    """
    return euclid_distance(center1, center2)


def sample_mean_for_the_intracluster_distance(clusters, centers):
    """
    d with a line
    """
    return sum([average_intercluster_distance(clusters[k], centers[k]) for k in range(len(clusters))]) / len(clusters)


def sample_variance_for_the_intracluster_distance(clusters, centers):
    """
    sigma_intra^2
    """
    return sum([(average_intercluster_distance(clusters[k], centers[k]) -
                 sample_mean_for_the_intracluster_distance(clusters, centers)) ** 2
                for k in range(len(clusters))]) / (len(clusters) - 1)


def sample_coefficient_of_variation_for_intracluster(clusters, centers):
    """
    C_intra
    """
    return (sample_variance_for_the_intracluster_distance(clusters, centers) /
            sample_mean_for_the_intracluster_distance(clusters, centers))


def sample_mean_for_the_intercluster_distance(centers):
    """
    D_with_line
    """
    D = 0
    for i in range(len(centers)):
        for j in range(len(centers)):
            if i != j:
                D += intercluster_distance_between_cluster(centers[i], centers[j])
    D /= len(centers) * (len(centers) - 1) / 2
    return D


def sample_variance_for_the_intercluster_distance(centers):
    """
    sigma_inter^2
    """
    D = 0
    for i in range(len(centers)):
        for j in range(len(centers)):
            if i != j:
                D += ((intercluster_distance_between_cluster(centers[i], centers[j]) -
                       sample_mean_for_the_intercluster_distance(centers)) ** 2)
    D /= len(centers) * (len(centers) - 1) / 2 - 1
    return D


def sample_coefficient_of_variation_for_intercluster(centers):
    """
    C_inter
    """
    return sample_variance_for_the_intercluster_distance(centers) / sample_mean_for_the_intercluster_distance(centers)


def beta_var(clusters, centers):
    return (sample_variance_for_the_intracluster_distance(clusters, centers) /
            sample_variance_for_the_intercluster_distance(centers))


def beta_CV(clusters, centers):
    return (sample_coefficient_of_variation_for_intracluster(clusters, centers) /
            sample_coefficient_of_variation_for_intercluster(centers))
