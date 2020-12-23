from _cluster_file_functions import extract_clusters, record_clusters

def fake_alignment(chain):
    aligned_chain = ''
    for i, c in enumerate(chain):
        aligned_chain += c + "%02d" % (i + 1)
    return aligned_chain

class CyclicAlignment:
    """
    Псевдовыравнивание без шаблона
    Равные состояния нумеруются одинаково внутри кластера
    Необходимо для визуализации с циклами
    """
    def __init__(self):
        self.dict_coding = {}

    def align(self, chain):
        aligned_chain = ''
        num = 0
        for letter in chain:
            if letter not in self.dict_coding:
                self.dict_coding[letter] = num
                num += 1
            aligned_chain += letter + str(self.dict_coding[letter])
        return aligned_chain


if __name__ == '__main__':

    cluster_source = 'data1/Clusters/patient_traces_test/KMeans8.txt'
    out_dir = cluster_source.split('.')[0] + '_fake_alignment.txt'

    clusters, patients = extract_clusters(cluster_source,
                                          sep='\t', eval_=False)

    for cluster in clusters:
        cyclic_alignment = CyclicAlignment()
        for i in range(len(cluster)):
            cluster[i] = cyclic_alignment.align(cluster[i])

    record_clusters(out_dir, clusters, patients, sep='\t')

