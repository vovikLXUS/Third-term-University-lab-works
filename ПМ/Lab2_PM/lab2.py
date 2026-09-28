import matplotlib.pyplot as plt
import numpy as np

def task1():
    # Створюємо вікно з двома підграфіками
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Графік а): x^2*y^2 + 2*x*y + x^3 = y^3  (Неявна крива)
    ax1 = axes[0]
    x_vals = np.linspace(-4, 4, 1000)
    y_vals = np.linspace(-4, 4, 1000)
    X, Y = np.meshgrid(x_vals, y_vals)

    # f(x, y) = 0
    F = X ** 2 * Y ** 2 + 2 * X * Y + X ** 3 - Y ** 3

    # Відображаємо ізолінію F = 0
    ax1.contour(X, Y, F, levels=[0], colors="purple", linewidths=2)

    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_title(r"а) $x^2 y^2 + 2xy + x^3 = y^3$", fontsize=13)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # -------------------------------------------------------------
    # Графік б): y = (2x^2 + 2x) / (x^2 + 1)  (Параметрична/Явна крива) (побудувати параметрично/переробить)
    # -------------------------------------------------------------
    ax2 = axes[1]
    # Генерація значень t від 0 до 2*pi
    t = np.linspace(0, 2 * np.pi, 1000)
    # Обчислення координат x та y згідно з варіантом "б"
    # Примітка: tg(t) у Python/NumPy записується як np.tan(t)
    x = np.tan(t)
    y = 2 * np.sin(t) ** 2 + np.sin(2 * t)
    # Побудова графіку
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.plot(x, y, color='b', label=r'Крива: $y = 2\sin^2(t) + \sin(2t)$, $x = tg(t)$')
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    plt.title('Параметрично задана крива (варіант б)')
    plt.legend()
    plt.grid(True)
    # Оскільки функція x = tg(t) має розриви і прямує до нескінченності,
    # варто обмежити осі графіка для його коректного та охайного відображення
    plt.xlim(-10, 10)
    plt.ylim(-1.5, 3.5)
    # Показ графіку
    plt.show()


def task2():
    # Задаємо параметри згідно із завданням 2
    a = 5 / 3
    b = -4 / 3
    # Генеруємо значення кута phi (від 0 до 2*pi)
    phi = np.linspace(0, 2 * np.pi, 1000)
    # Обчислюємо r за формулою (f(phi) = sin(phi))
    r = a / (1 + b * np.sin(phi))
    # Обчислюємо координати x і y (як у вашому прикладі)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    # 1. Графік у полярних координатах
    plt.figure(figsize=(8, 6))
    plt.polar(phi, r, color='blue', label="r = (5/3) / (1 - (4/3) * sin(phi))")
    # Підписуємо малюнок
    plt.title('Крива другого порядку в полярній системі координат')
    plt.legend(loc='upper right')
    plt.grid(True)
    # Автоматичне налаштування відстаней між елементами
    plt.tight_layout()
    # Відображення графіка
    plt.show()



if __name__ == "__main__":
    task1()
    task2()