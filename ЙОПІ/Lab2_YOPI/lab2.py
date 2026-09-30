import math


def factorial(number: int) -> int:
    """Calculate factorial."""
    if number < 0:
        raise ValueError("Factorial of a negative number is undefined.")
    if number == 0 or number == 1:
        return 1
    res = 1
    for i in range(2, number + 1):
        res *= i
    return res


def combination(n: int, k: int) -> int:
    """Calculate combinations C(n, k)."""
    if k < 0 or k > n:
        return 0
    return factorial(n) // (factorial(k) * factorial(n - k))


def bernoulli(n: int, k: int, p: float) -> float:
    """Calculate probability using Bernoulli formula."""
    if k < 0 or k > n:
        return 0.0
    q = 1.0 - p
    return combination(n, k) * (p ** k) * (q ** (n - k))


def poisson(k: int, lam: float) -> float:
    """Calculate probability using Poisson formula."""
    if k < 0:
        return 0.0
    return (lam ** k * math.exp(-lam)) / factorial(k)


def phi_density(x: float) -> float:
    """Differential Laplace function."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def local_moivre_laplace(n: int, k: int, p: float) -> float:
    """Local Moivre-Laplace theorem."""
    q = 1.0 - p
    variance = n * p * q
    std_dev = math.sqrt(variance)
    x = (k - n * p) / std_dev
    return phi_density(x) / std_dev


def laplace_function(x: float) -> float:
    """
    Laplace integral function Phi_0(x).
    Phi_0(x) = (1 / sqrt(2 * pi)) * integral_0^x exp(-t^2 / 2) dt.
    """
    return 0.5 * math.erf(x / math.sqrt(2.0))


def integral_moivre_laplace(n: int, k1: int, k2: int, p: float) -> float:
    """Integral Moivre-Laplace theorem for P_n(k1 <= k <= k2)."""
    q = 1.0 - p
    std_dev = math.sqrt(n * p * q)
    x1 = (k1 - n * p) / std_dev
    x2 = (k2 - n * p) / std_dev
    return laplace_function(x2) - laplace_function(x1)


def most_probable_successes(n: int, p: float) -> int:
    """Find the most probable number of successes (mode)."""
    upper = n * p + p
    return math.floor(upper)


def solve_tasks():
    print("=" * 60)
    print("SOLUTIONS FOR LAB WORK 2")
    print("=" * 60)

    # Task 1: Bernoulli formula (n = 5, k = 3, p = 0.2)
    p1 = bernoulli(5, 3, 0.2)
    print(f"Task 1: P = {p1:.4f}")

    # Task 2: Bernoulli trials (n = 5, p = 0.8)
    p2_a = bernoulli(5, 4, 0.8)
    p2_b = bernoulli(5, 4, 0.8) + bernoulli(5, 5, 0.8)
    print(f"Task 2: a) P(k=4) = {p2_a:.4f}; b) P(k>=4) = {p2_b:.4f}")

    # Task 3: Local Moivre-Laplace theorem (n = 400, k = 80, p = 0.2)
    p3_ml = local_moivre_laplace(400, 80, 0.2)
    p3_exact = bernoulli(400, 80, 0.2)
    print(f"Task 3: P_ML = {p3_ml:.4f} (P_exact = {p3_exact:.4f})")

    # Task 4: Poisson distribution (n = 100000, p = 0.0001, k = 5)
    lam4 = 100000 * 0.0001
    p4_poiss = poisson(5, lam4)
    print(f"Task 4: P_Poisson = {p4_poiss:.4f}")

    # Task 5: Integral Moivre-Laplace theorem (n = 600, p = 0.4, k in [228, 252])
    p5_ml = integral_moivre_laplace(600, 228, 252, 0.4)
    print(f"Task 5: P_ML = {p5_ml:.4f}")

    # Task 6: Most probable number of clients (n = 100, p = 0.4)
    k0_6 = most_probable_successes(100, 0.4)
    p6_ml = local_moivre_laplace(100, k0_6, 0.4)
    p6_exact = bernoulli(100, k0_6, 0.4)
    print(f"Task 6: k0 = {k0_6}, P = {p6_ml:.4f} (P_exact = {p6_exact:.4f})")

    # Task 7: Integral Moivre-Laplace theorem (n = 4000, p = 0.04, k <= 170)
    p7_ml = integral_moivre_laplace(4000, 0, 170, 0.04)
    print(f"Task 7: P_ML = {p7_ml:.4f}")

    # Task 8: Local Moivre-Laplace theorem (n = 10000, p = 0.5, k = 5000)
    p8_ml = local_moivre_laplace(10000, 5000, 0.5)
    print(f"Task 8: P_ML = {p8_ml:.4f}")

    # Task 9: Poisson distribution (n = 1000, p = 0.002, k = 5)
    lam9 = 1000 * 0.002
    p9_poiss = poisson(5, lam9)
    print(f"Task 9: P_Poisson = {p9_poiss:.4f}")

    # Task 10: Most probable number of successes (n = 150, p_correct = 0.97)
    k0_10 = most_probable_successes(150, 0.97)
    p10_exact = bernoulli(150, k0_10, 0.97)
    print(f"Task 10: k0 = {k0_10}, P = {p10_exact:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    solve_tasks()

