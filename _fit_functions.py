from alignment import align_word


def fit_alignment(cluster: list, template: str, absolute=False):
    """
    Целевая функция выравнивания. Выдаёт число
    невыравненных последовательностей под заданный шаблон
    :param cluster: последовательности кластера
    :param template: шаблон
    :param absolute: если True возвращает количество
    выравненных последовательностей,
    иначе относительное число невыравненных
    последовательностей и все такие последовательности
    :return: см. выше
    """
    value = 0
    no_alignment_word = []
    for word in cluster:
        if align_word(word, template, print_execution=False):
            value += 1
        else:
            no_alignment_word.append(word)

    if absolute:
        return value
    else:
        return (len(cluster) - value) / len(cluster), no_alignment_word


def fit_length(template: str, max_len: int):
    """
    Целевая функция длины шаблона
    :param template: шаблон
    :param max_len: максимально возможная длина шаблона
    :return: относительная длина шаблона
    """
    return len(template) / max_len
