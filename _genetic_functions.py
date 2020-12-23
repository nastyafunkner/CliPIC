import numpy
import random


def generate_initial_population(size: int, alphabet: str, mutation_probability: list,
                                min_len=0, max_len=10, approximation=None):
    """
    Генерирует изначальную популяция осыбей
    :param size: размер популяции
    :param alphabet: всевозможные состояния для генерации последовательностей
    :param mutation_probability: вероятность мутации
    :param min_len: мин. длина пос-ти
    :param max_len: макс. длина пос-ти
    :param approximation: если заданна пос-ть, 
    генерируются пос-ти похожие на данную, 
    иначе генерируются абсолютно случайные пос-ти
    :return: популяция
    """
    population = []
    while len(population) < size:
        if approximation:
            word = approximation
            for i in range(int(len(word) / 2)):
                word = mutation(word, mutation_probability, alphabet)
        else:
            word = ''.join([random.choice(alphabet)
                            for _ in range(random.randint(min_len, max_len))])
        if min_len < len(word) < max_len and word not in population:
            population.append(word)
    return population


def mutation(sequence: str, probabilities: list, alphabet: str):
    """
    Изменяет пос-ть с помощью трёх мутаций: инсерции, делеции и замены
    :param sequence: пос-ть
    :param probabilities: вероятность каждой из трёх видов мутации
    :param alphabet: всевозможные состояния
    :return: мутировавшая пос-ть
    """
    chosen_mutation = numpy.random.choice([substitution, insertion, deletion, None],
                                          p=probabilities + [1 - sum(probabilities)])
    if chosen_mutation:
        return chosen_mutation(sequence, alphabet)
    else:
        return sequence


def substitution(sequence, alphabet):
    loc = random.randint(0, len(sequence) - 1)
    new_sequence = sequence[:loc] + random.choice(alphabet) + sequence[loc + 1:]
    return new_sequence


def insertion(sequence, alphabet):
    loc = random.randint(0, len(sequence) - 1)
    new_sequence = sequence[:loc] + random.choice(alphabet) + sequence[loc:]
    return new_sequence


def deletion(sequence, alphabet):
    if len(sequence) < 3:
        return sequence
    else:
        loc = random.randint(0, len(sequence) - 1)
        new_sequence = sequence[:loc] + sequence[loc + 1:]
        return new_sequence


def crossover(mother: str, father: str):
    """
    Реализован алгоритм кроссовера
    :param mother: первая пос-ть
    :param father: вторая пос-ть
    :return: новая пос-ть, полученная из двух предыдущих
    """
    if len(mother) <= 2 or len(father) <= 2:
        return mother if len(mother) >= len(father) else father
    while True:
        loc_mother = random.randint(1, len(mother) - 1)
        loc_father = random.randint(1, len(father) - 1)
        children = [mother[:loc_mother] + father[loc_father:], father[:loc_father] + mother[loc_mother:]]
        choice = random.choice([0, 1])
        if choice:
            if 2 < len(children[choice]):
                return children[choice]
            else:
                return children[0]
        else:
            if 2 < len(children[choice]):
                return children[choice]
            else:
                return children[1]
