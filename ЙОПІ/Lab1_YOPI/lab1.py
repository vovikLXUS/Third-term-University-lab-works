def factorial(number: int) -> int:
    """Calculate factorial."""
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


def total_probability(
        probabilities_of_hypotheses: list[float],
        conditional_probs: list[float]
) -> float:
    """Law of total probability."""
    return sum(
        p * c for p, c in zip(
            probabilities_of_hypotheses,
            conditional_probs
        )
    )


def bayes_theorem(
        prior_probs: list[float],
        conditional_probs: list[float],
        target_index: int
) -> float:
    """Bayes' theorem for a given hypothesis."""
    p_total = total_probability(prior_probs, conditional_probs)
    p_target = prior_probs[target_index] * conditional_probs[target_index]
    return p_target / p_total


def solve_tasks():
    # Task 1
    black, brown, red, blue = 40, 26, 22, 12
    total_shoes = black + brown + red + blue
    p1 = (red + blue) / total_shoes
    print(f"Task 1: P = {p1:.4f}")

    # Task 2
    total_emp, consultants, to_choose = 10, 8, 2
    non_consultants = total_emp - consultants
    p_1 = combination(non_consultants, to_choose) / combination(total_emp, to_choose)
    p2 = 1 - p_1
    print(f"Task 2: P = {p2:.4f}")

    # Task 3
    total_mgr, relatives, to_choose_m = 10, 2, 3
    non_relatives = total_mgr - relatives
    p_no_relatives = combination(non_relatives, to_choose_m) / combination(total_mgr, to_choose_m)
    p3 = 1 - p_no_relatives
    print(f"Task 3: P = {p3:.4f}")

    # Task 4
    p_deps = [0.15, 0.25, 0.2, 0.1]
    p4 = 1 - sum(p_deps)
    print(f"Task 4: P = {p4:.4f}")

    # Task 5
    total_tracks, trains = 120, 80
    # Probability that both adjacent tracks are occupied
    p5 = (trains / total_tracks) * ((trains - 1) / (total_tracks - 1))
    print(f"Task 5: P = {p5:.4f}")

    # Task 6
    p_std, p_1st_given_std = 0.9, 0.8
    p6 = p_std * p_1st_given_std
    print(f"Task 6: P = {p6:.4f}")

    # Task 7
    students_prob = [3 / 10, 4 / 10, 2 / 10, 1 / 10]  # Hypotheses
    questions_total, asked = 20, 3
    # Probabilities of answering 3 questions for each group
    cond_probs = [
        combination(20, asked) / combination(questions_total, asked),
        combination(16, asked) / combination(questions_total, asked),
        combination(10, asked) / combination(questions_total, asked),
        combination(5, asked) / combination(questions_total, asked)
    ]
    p7_a = bayes_theorem(students_prob, cond_probs, target_index=0)  # excellent
    p7_b = bayes_theorem(students_prob, cond_probs, target_index=3)  # poor
    print(f"Task 7: a) P(excellent) = {p7_a:.4f}; b) P(poor) = {p7_b:.4f}")

    # Task 8
    lines_prob = [0.4, 0.3, 0.3]
    std_probs = [0.9, 0.95, 0.95]
    p8 = total_probability(lines_prob, std_probs)
    print(f"Task 8: P = {p8:.4f}")

    # Task 9
    diseases_prob = [0.4, 0.3, 0.3]  # pneumonia, peritonitis, tonsillitis
    recovery_probs = [0.8, 0.7, 0.85]
    p9 = bayes_theorem(diseases_prob, recovery_probs, target_index=1)  # peritonitis (index 1)
    print(f"Task 9: P = {p9:.4f}")

    # Task 10
    specialists_prob = [0.3, 0.7]  # high, medium
    reliability_probs = [0.9, 0.8]
    p10 = bayes_theorem(specialists_prob, reliability_probs, target_index=0)  # high (index 0)
    print(f"Task 10: P = {p10:.4f}")


if __name__ == "__main__":
    solve_tasks()
