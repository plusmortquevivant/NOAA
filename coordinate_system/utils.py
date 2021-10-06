"""
Методы, которым не придумано лучшего места
"""

from typing import Union, Any, Tuple
import numpy as np
from .types import FloatsArray, BoolsArray, PositionsArray, Position, Position1

ACCURACY = 1e-10


def dot_rows(a: np.ndarray, b: np.ndarray) -> Union[np.ndarray, float]:
    """
    Рассчитать скалярное произведение для массивов векторов,представленных numpy массивами. Должно быть возможно 'a * b'

    Происходит вычисление результата операции 'a * b' и суммирование вдоль последнего индекса результата. Работает для
    всех тех случаев, когда произведение определено. Допустимо, чтобы один из операндов был одним вектором
    (как и одномерный массив, так и двумерный массив, у которого размерность по первому измерению равна 1)
    Один из них операндов может не быть numpy массивом, если определено 'a * b'.

    Если a и b одномерные массивы(представляют собой просто вектора), то результат будет float, иначе результат будет
    представлять собой np.ndarray

    :param a: первый массив векторов
    :param b: второй массив векторов
    :return: результат скалярного умножения. Размерность зависит от размеров операндов.
    """
    c = a * b
    return np.sum(c, axis=len(c.shape) - 1)


def dot_rows_fast_3D(a: np.ndarray, b: np.ndarray) -> Union[np.ndarray, float]:
    """
    Скалярное произведение для массивов трехмерных векторов,представленных numpy массивами. Должно быть возможно 'a * b'

    Происходит вычисление результата операции 'a * b' и суммирование вдоль последнего индекса результата.
    Требуется, чтобы размерность последней оси была равна 3!!!
    всех тех случаев, когда произведение определено. Допустимо, чтобы один из операндов был одним вектором
    (как и одномерный массив, так и двумерный массив, у которого размерность по первому измерению равна 1)
    Один из них операндов может не быть numpy массивом, если определено 'a * b'.

    Если a и b одномерные массивы(представляют собой просто вектора), то результат будет float, иначе результат будет
    представлять собой np.ndarray

    :param a: первый массив векторов
    :param b: второй массив векторов
    :return: результат скалярного умножения. Размерность зависит от размеров операндов.
    """
    c = a * b
    if c.shape[-1] != 3:
        raise ValueError("Передан не массив трехмерных векторов")
    res = c[..., 0] + c[..., 1] + c[..., 2]
    return res


def dot_rows_quadratic_forms(a: np.ndarray, quadratic_form: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Вычисление построчного произведения

    Может быть ускорена на 10-15% путем хака, который плохо ляжет в логику

    :param a: массив первых векторов(N векторов размерности 3)
    :param quadratic_form: массив квадратичных форм(N квадратичных форм размерности 3х3)
    :param c: массив вторых векторов(N векторов размерности 3)
    :return: Для каждой тройки результат a' * quadratic_form * c. N чисел
    """
    return np.einsum('ij,ijk,ik->i', a, quadratic_form, c)


def matrix_multiple_batch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Вычисление построчного произведения

    Может быть ускорена на 10-15% путем хака, который плохо ляжет в логику

    :param A: массив матриц(N матриц размерности k * l)
    :param B: массив матриц(N матриц размерности l * m)
    :return: Для каждой пары матрик результат A * B. N матриц размера k * m
    """
    return np.einsum('ijk,ikm->ijm', A, B)


def dot_rows_any_type(a: Any, b: Any) -> Union[np.ndarray, float]:
    """
    Рассчитать скалярное произведение для массивов векторов. Должно быть возможно 'np.array(a) * np.array(b)'

    Происходит вычисление результата операции 'a * b' и суммирование вдоль последнего индекса результата. Работает для
    всех тех случаев, когда произведение определено. Допустимо, чтобы один из операндов был одним вектором
    (как и одномерный массив, так и двумерный массив, у которого размерность по первому измерению равна 1)

    Если a и b одномерные массивы(представляют собой просто вектора), то результат будет float, иначе результат будет
    представлять собой np.ndarray

    :param a: первый массив векторов
    :param b: второй массив векторов
    :return: результат скалярного умножения. Размерность зависит от размеров операндов.
    """
    return dot_rows(np.array(a), np.array(b))


def norm_rows(a: np.ndarray) -> Union[np.ndarray, float]:
    """
    Рассчитать нормы массивов векторов, представленных numpy массивом.

    Если a одномерный массив(представляет собой просто вектор), то результат будет float, иначе результат будет
    представлять собой np.ndarray

    :param a: массив векторов
    :return: возвращает норму для каждого вектора
    """
    return np.sqrt(dot_rows(a, a))


def norm_rows_fast_3D(a: np.ndarray) -> Union[np.ndarray, float]:
    """
    Рассчитать нормы массивов трехмерных векторов, представленных numpy массивом.

    Размерность вдоль последнего измерения должна быть равна 3.
    Если a одномерный массив(представляет собой просто вектор), то результат будет float, иначе результат будет
    представлять собой np.ndarray

    :param a: массив векторов
    :return: возвращает норму для каждого вектора
    """
    return np.sqrt(dot_rows_fast_3D(a, a))


def norm_rows_any_type(a: Any) -> Union[np.ndarray, float]:
    """
    Рассчитать норму для каждого вектора в массиве векторов. Выполняется приведение к numpy массиву.

    Если a одномерный массив(представляет собой просто вектор), то результат будет float, иначе результат будет
    представлять собой np.ndarray

    :param a: массив векторов
    :return: возвращает норму для каждого вектора
    """
    return norm_rows(np.ndarray(a))


def get_quadratic_solution(quad_term: FloatsArray,
                           lin_term: FloatsArray,
                           const_term: FloatsArray,
                           accuracy: float = ACCURACY,
                           calculate_roots: bool = True) -> Tuple[BoolsArray, BoolsArray, FloatsArray]:
    """
    Решение квадратного уравнения(если коэффициент при квадратичном члене равен нулю, то линейного)

    :param quad_term: массив коэффициентов при квадратичном члене
    :type quad: float

    :param lin_term: массив коэффициентов при линейном члене
    :type lin_term: float

    :param const_term: массив свободных членов
    :type const_term: float

    :param accuracy: точность на основание которой принимается решение квадратное ли это уравнение.
    :param calculate_roots: вычислять ли корни уравнения.Если установлено в False то вместо корней будет возвращен мусор

    :return: Индекс какие именно уравнения имели решение. True на соответствующей позиции, если было получено решение
             Индекс линейных уравнений. True, если уравнение линейное (quad_term < accuracy lin_term > accuracy)
             Корни для решенных уравнений. Вычеркнуты те строчки, для которых нет решения. Для каждого случая
             приводится 2 решения. Для линейного уравнения - совпадают.
    """

    discriminant = lin_term ** 2 - 4 * quad_term * const_term
    linear_case = quad_term < accuracy
    have_linear_solution = linear_case * (lin_term > accuracy)

    have_quadratic_solution = (discriminant > 0) * (~linear_case)
    have_solution = have_quadratic_solution + have_linear_solution
    solution_count = have_solution.sum()

    solution_index_linear = have_linear_solution[have_solution]
    solution_index_quadratic = ~solution_index_linear
    solutions = np.zeros((solution_count, 2))

    if calculate_roots:
        solutions[solution_index_linear] = -(const_term / lin_term)[have_linear_solution, np.newaxis]

        discriminant_sqrt = np.sqrt(discriminant[have_quadratic_solution])
        lin_quad_sol = lin_term[have_quadratic_solution]
        quad_quad_sol = quad_term[have_quadratic_solution] * 2
        solutions[solution_index_quadratic, 0] = -(lin_quad_sol + discriminant_sqrt) / quad_quad_sol
        solutions[solution_index_quadratic, 1] = -(lin_quad_sol - discriminant_sqrt) / quad_quad_sol

    return have_solution, have_linear_solution, solutions


def generate_orthonormal_basis_by_one_direction(direction: PositionsArray) -> \
        Tuple[PositionsArray, PositionsArray, PositionsArray]:
    """
    Возвращает ортонормированный базис, третья ось которого совпадает по направлению с direction

    :param direction: направление третьей оси базиса
    :return: базис
    """
    e3 = direction / norm_rows(direction)[:, np.newaxis]

    index_zero = e3[:, 0] == 0
    index_non_zero = ~index_zero
    e1 = np.zeros(e3.shape)

    if len(index_zero) > 0:  # случай когда заданное направление перпендикулярно оси абцисс
        e1[index_zero] = [1, 0, 0]
    if len(index_non_zero) > 0:
        e1[index_non_zero] = [0, 1, 0]
        e1[index_non_zero, 0] = -e3[index_non_zero, 1] / e3[index_non_zero, 0]
    e1 = e1 / norm_rows(e1)[:, np.newaxis]
    e2 = np.cross(e3, e1)
    return e1, e2, e3

def generate_sphere(position: Position,
                    radius: float,
                    discretization_longitude: int = 98,
                    discretization_latitude: int = None):
    """
    Генерирует точки, принадлежащие сфере заданного радиуса

    :param position: координаты центра сферы
    :param radius: радиус сферы
    :param discretization_longitude: количество точек для дискретизации долготы
    :param discretization_latitude: количество точек для дискретизации широты
    :return: точки, лежащие на сфере
    """
    if discretization_latitude is None:
        discretization_latitude = discretization_longitude // 2

    longitude = np.linspace(0, 2 * np.pi, discretization_longitude - 2, False)
    latitude = np.linspace(- np.pi / 2, np.pi / 2, discretization_latitude, True)

    grid = np.array(np.meshgrid(longitude, latitude))

    longitude = grid[0].reshape(np.prod(grid[0].shape))
    latitude = grid[1].reshape(np.prod(grid[1].shape))

    lon_1 = longitude[343]
    lat_1 = latitude[343]

    lat_2 = np.arcsin(np.cos(lat_1) * np.sin(lon_1))
    lon_2 = - np.arcsin(np.sin(lat_1) / np.cos(lat_2)) + np.pi

    cartesian = np.zeros(len(longitude), dtype=Position1)
    cartesian[:, 2] = np.sin(latitude)
    cartesian[:, 0] = np.cos(latitude)
    cartesian[:, 1] = np.cos(latitude) * np.sin(longitude)
    cartesian[:, 0] = np.cos(latitude) * np.cos(longitude)
    cartesian *= radius
    cartesian += position

    return cartesian
