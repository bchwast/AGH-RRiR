from math import sqrt
import numpy as np
from matplotlib import pyplot as plot


# całkowanie metodą kwadratury Gaussa-Legendre'a
def gaussian_quad(f, a, b):
    x1 = -1 / sqrt(3)
    x2 = 1 / sqrt(3)

    w1 = 1
    w2 = 1

    c1 = (b - a) / 2
    c2 = (b + a) / 2

    u1 = c1 * x1 + c2
    u2 = c1 * x2 + c2

    return c1 * (w1 * f(u1) + w2 * f(u2))


# funkcja E(x)
def e(x):
    if 0 <= x <= 1:
        return 3
    if 1 < x <= 2:
        return 5
    return 0


# punkt podziału
def x_i(i, n):
    return 2.0 * i / n


# i-ty element
def e_i(i, x, n):
    if x_i(i - 1, n) <= x <= x_i(i, n):
        return (x - x_i(i - 1, n)) / (2 / n)
    if x_i(i, n) < x <= x_i(i + 1, n):
        return -1 * (x - x_i(i + 1, n)) / (2 / n)
    return 0


# pochodna i-tego elementu
def e_i_d(i, x, n):
    if x_i(i - 1, n) <= x <= x_i(i, n):
        return n / 2
    if x_i(i, n) < x < x_i(i + 1, n):
        return -1 * n / 2
    return 0


# element macierzy związany z funkcją B
def b_ei_ej(i, j, n):
    a = max(0, x_i(i - 1, n), x_i(j - 1, n))
    b = min(2, x_i(i + 1, n), x_i(j + 1, n))
    f = lambda x: e(x) * e_i_d(i, x, n) * e_i_d(j, x, n)
    return gaussian_quad(f, a, b) - 3 * e_i(i, 0, n) * e_i(j, 0, n)


# element macierzy związany z funkcją L
def l_ei(i, n):
    return -30 * e_i(i, 0, n)


# wyliczanie wartości funkcji u
def u(alphas, x, n):
    result = 0
    for i in range(len(alphas)):
        result += alphas[i] * e_i(i, x, n)
    return result


# wypełnianie macierzy, rozwiązywanie układu równań, obliczanie wartości funkcji u
# wykres u(x) i wykres funkcji bazowych
def solver(n):
    M = []
    B = []

    # wypełniam macierz M
    for i in range(n):
        M.append([])
        for j in range(n):
            M[i].append(b_ei_ej(j, i, n))

    # wypełniam macierz B
    for i in range(n):
        B.append(l_ei(i, n))

    solution = np.linalg.solve(np.array(M), np.array(B))

    values = [u(solution, x_i(i, 100), n) for i in range(101)]

    x = np.linspace(0, 2, n)
    for j in range(n):
        plot.plot(x, [e_i(j, x_i(i, n), n) for i in range(n)])
    plot.title("Wykres funkcji bazowych")
    plot.savefig("base_functions.pdf")
    plot.show()


    x = np.linspace(0, 2, 101)
    plot.plot(x, values)
    plot.title("Wykres przybliżonej funkcji y = u(x)")
    plot.savefig("u(x).pdf")
    plot.show()


solver(3)
