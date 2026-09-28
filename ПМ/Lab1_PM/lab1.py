import numpy as np
import sympy as sp


def task1():
    A = np.array(
        [
            [2, 1, 3],
            [-3, 0, -1],
            [4, 2, -1]
        ]
    )

    # Створення квадратної матриці 3х3
    # .eye() створює матрицю типу float, 'dtype=int' - явно вказує на цільний тип чисел
    I = np.eye(3, dtype=int)

    f_A = -2 * np.linalg.matrix_power(A, 2) + 8 * A - 6 * I
    print("\n=== Task 1 ===")
    print(f"Result:\n{f_A}")


def task2():
    A = np.array(
        [
            [4, 3, 6],
            [5, 4, 7],
            [6, 5, 7]
        ]
    )
    B = np.array(
        [
            [2, 1, 5],
            [0, 0, -1]
        ]
    )

    # .linalg.inv() = А^-1, іншими словами обернена матриця А,
    # символ '@' використовується для множення матриць
    x = B @ (np.linalg.inv(A))
    x = np.round(x).astype(int)

    print("\n=== Task 2 ===")
    print(f"Result:\n{x}")


def task3():
    A = np.array(
        [
            [1, 2, 1, 0],
            [0, 1, 3, 1],
            [4, 0, 1, 1],
            [1, 1, 0, 5]
        ],
        dtype=float
    )
    b = np.array([8, 15, 11, 23], dtype=float)

    # Розширена матриця
    A_aug = np.column_stack((A, b))

    # Обчислення рангів матриць, якщо рівні - система сумісна і має один розв'язок
    rank_A = np.linalg.matrix_rank(A)
    rank_A_aug = np.linalg.matrix_rank(A_aug)

    print("\n=== Task 3 ===")
    print(f"Rank of A matrix: {rank_A}")
    print(f"Rank of A augmented: {rank_A_aug}")

    # Перевірка сумісності
    if rank_A == rank_A_aug:
        x = np.linalg.solve(A, b)
        x_rounded = np.round(x).astype(int)
        print(
            f"Consistent system. Unique solution: x1={x_rounded[0]}, "
            f"x2={x_rounded[1]}, x3={x_rounded[2]}, x4={x_rounded[3]}"
        )
    else:
        print("System is not consistent (no solution).")


def task4():
    a, b = sp.symbols('a b')
    expression = (3 * a + 2 * b) * (-a + 3 * b)
    # Розширення виразу
    expanded_expression = sp.expand(expression)
    # Довжини векторів та кут між ними
    a_mag = 4 * np.sqrt(2)
    b_mag = 3
    angle = np.pi / 4

    # Обчислення скалярного добутку
    scalar_product = a_mag * b_mag * np.cos(angle)

    # Підстановка відповідних значень у вираз
    result = expanded_expression.subs(
        {
            a**2: a_mag**2,
            b**2: b_mag**2,
            a*b: scalar_product
        }
    )

    print("\n=== Task 4(a) ===")
    print(f"Expanded expression: {expanded_expression}")
    print(f"Result after substitution: {result}")

    result = np.sqrt(4 * (a_mag**2) + 12 * scalar_product + 9 * (b_mag**2))
    print("\n=== Task 4(b) ===")
    print(f"Result for |2a + 3b|: {result:.6f}")


def task5():
    A = np.array([0, -3, 5])
    B = np.array([-3, -1, 1])
    C = np.array([2, -5, 2])
    D = np.array([4, 3, 6])

    AB = B - A
    AC = C - A
    AD = D - A

    cross_prod = np.cross(AB, AC)

    S_abc = 0.5 * np.linalg.norm(cross_prod)

    mixed_prod = np.dot(cross_prod, AD)
    V_abcd = (1 / 6) * abs(mixed_prod)

    print("\n=== Task 5 ===")
    print(f"Area of a face of ABC: {S_abc:.4f}")
    print(f"Volume of a piramid: {V_abcd:.1f}")
