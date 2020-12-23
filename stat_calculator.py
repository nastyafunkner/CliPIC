import pandas as pd

from _cluster_file_functions import extract_clusters




def cluster_report(cluster_file,
                   alignment_file):
    clusters, cases = extract_clusters(cluster_file, eval_=False)
    df = pd.DataFrame(columns=['case_id', 'cluster', 'sequence'])
    cases_col = []
    cluster_col = []
    sequence_col = []
    for i, (cluster, ids) in enumerate(zip(clusters, cases)):
        cluster_col += [i] * len(cluster)
        sequence_col += cluster
        cases_col += ids

    df['case_id'] = list(map(str, cases_col))
    df['cluster'] = cluster_col
    df['sequence'] = sequence_col
    
    
    clusters, cases = extract_clusters(alignment_file, eval_=False)
    aligned = []
    for ids in cases:
        aligned += ids

    df['aligned'] = df.case_id.isin(aligned)
    df['outliers'] = ~df.aligned
    df['len'] = df.sequence.apply(len)

    summed = df.groupby('cluster').sum()[['aligned', 'outliers']]
    mean_val = df.groupby('cluster').mean()[['len']].round(2)

    base_report = pd.merge(summed,
                           mean_val,
                           left_index=True,
                           right_index=True)

    report = base_report.copy()

    report['size'] = (report.aligned + report.outliers).apply(int)
    report['outliers'] = report.outliers.apply(int)
    

    col_order = ['size']
    for col in ['outliers']:
        proc_name = col + '_%'
        report[proc_name] = (report[col] / report['size'] * 100).round(2)
        col_order.append(col)
        col_order.append(proc_name)

    col_order += ['len']
    report = report[col_order].rename(
        columns={
            'len': 'mean_seq_len'
        }
    )

    report.to_csv(alignment_file.replace('.txt', '_report.csv'),
                  sep=';',
                  index=False)

    return report

if __name__ == '__main__':
    CLUSTERS_PATH = r'data1/Clusters/patient_traces_test/KMeans6.txt'
    ALIGNMENT_PATH = r'data1/Clusters/patient_traces_test/KMeans6_full_alignment.txt'
    print(cluster_report(CLUSTERS_PATH, ALIGNMENT_PATH))
