# Clinical Pathway Identification and Clustering (CliPIC)

Проект посвящен кластеризации и визуализации клинических путей (КП) пациентов.
Может быть использован для кластеризации и визуализации любых других последовательностей состояний (например, последовательность действий пользователей социальной сети и др.)

## Getting Started

Необходимы дополнительные модули питон: sklearn, numpy, matplotlib, graphviz.
Также для визуализации необходим Graphviz.
Скачать можно тут: https://www.graphviz.org/download/

Запускать проект можно с помощью файла main или ноутбуков (**use_clipic**.ipynb, **use_clipic_fake_alignment**.ipynb).
Необходимо указать все начальные и конечные директории в main.
Также проект можно запускать поскриптово:
1. clustering.py - проводит кластеризацию КП и позволяет определить число кластеров
2. выравнивание:
    1. fake_alignment.py - проводит выравнивание КП внутри каждого кластера. Позволяет получить ЦИКЛИЧЕСКИЕ обобщенные пути для каждого кластера
    2. alignment.py - проводит выравнивание КП внутри каждого кластера. Необходимы шаблоны для выравнивания. Позволяет получить ЛИНЕЙНЫЕ обобщенные пути для каждого кластера
3. cluster_visualization.py - визуализирует КП на основе выровненных последовательностей  

### Больше о работе методов
https://www.sciencedirect.com/science/article/pii/S1877050917323918
