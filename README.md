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

### Использовалось в статьях:
* https://www.sciencedirect.com/science/article/pii/S153204641830087X
* https://link.springer.com/chapter/10.1007/978-3-030-01129-1_27
* https://www.hindawi.com/journals/complexity/2018/5870987/
* https://books.google.ru/books?hl=en&lr=&id=m2OmDwAAQBAJ&oi=fnd&pg=PA150&dq=info:S3jojYdGT6QJ:scholar.google.com&ots=yRso4xmrBH&sig=20rIxr2rWAwij50gGjDG1z8d0YU&redir_esc=y#v=onepage&q&f=false
* https://www.sciencedirect.com/science/article/pii/S1877050918316661
