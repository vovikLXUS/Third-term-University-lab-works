"""
Лабораторна робота № 2
Дисципліна: Прикладна математика
Тема: Графічні можливості Matplotlib. Побудова двовимірних графіків.
Виконав: Дацишин Володимир, група ІПЗ-23
Варіант: 8
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from scipy.optimize import brentq

# Створення папки для збереження зображень
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


# ==============================================================================
# ЗАВДАННЯ 1 (Загальне): РІЗНОКОЛЬОРОВІ ДІАГРАМИ (Варіант 8)
# ==============================================================================

def task1_1_stacked_bar(save=True, show=False):
    """
    1.1. Стовпчаста діаграма:
    - Кількість груп: Варіант + 5 = 8 + 5 = 13 груп.
    - Структура стовпчика: 4 різні кольорові сегменти (компоненти).
    - Підписи осей, заголовок, легенда.
    """
    num_groups = 8 + 5  # 13 груп
    np.random.seed(42)  # Фіксація випадковості для відтворюваності

    categories = [f"Група {i + 1}" for i in range(num_groups)]
    # Генеруємо значення для 4 сегментів кожного стовпчика
    seg1 = np.random.randint(10, 30, size=num_groups)
    seg2 = np.random.randint(15, 35, size=num_groups)
    seg3 = np.random.randint(10, 25, size=num_groups)
    seg4 = np.random.randint(5, 20, size=num_groups)

    fig, ax = plt.subplots(figsize=(12, 7))

    width = 0.6
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

    # Побудова складених стовпчиків (stacked bars)
    b1 = ax.bar(categories, seg1, width, label='Сегмент A (Базовий)', color=colors[0], edgecolor='black', linewidth=0.5)
    b2 = ax.bar(categories, seg2, width, bottom=seg1, label='Сегмент B (Операційний)', color=colors[1], edgecolor='black', linewidth=0.5)
    b3 = ax.bar(categories, seg3, width, bottom=seg1 + seg2, label='Сегмент C (Інвестиційний)', color=colors[2], edgecolor='black', linewidth=0.5)
    b4 = ax.bar(categories, seg4, width, bottom=seg1 + seg2 + seg3, label='Сегмент D (Резервний)', color=colors[3], edgecolor='black', linewidth=0.5)

    # Підписи сумарних значень над стовпчиками
    totals = seg1 + seg2 + seg3 + seg4
    for i, total in enumerate(totals):
        ax.text(i, total + 1.2, str(total), ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title("Складена стовпчаста діаграма з 4-ма сегментами (13 груп, Варіант 8)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Групи спостережень", fontsize=12, labelpad=10)
    ax.set_ylabel("Сумарне значення показника", fontsize=12, labelpad=10)
    ax.set_ylim(0, max(totals) * 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(IMAGES_DIR, "task1_1_stacked_bar.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def task1_2_pie_chart(save=True, show=False):
    """
    1.2. Кругова діаграма з секторами:
    - Кількість секторів: Варіант + 5 = 8 + 5 = 13 секторів.
    - Вийняті сектори: кожен другий сектор виділений (зміщений).
    - Підписи з відсотками та назвами, заголовок.
    """
    num_sectors = 8 + 5  # 13 секторів
    np.random.seed(88)

    labels = [f"Сектор {i + 1}" for i in range(num_sectors)]
    sizes = np.random.uniform(15, 60, size=num_sectors)

    # Виймаємо кожен другий сектор (парні індекси зміщуємо на 0.08)
    explode = [0.08 if i % 2 == 1 else 0.0 for i in range(num_sectors)]

    # Використовуємо колірну палітру
    colors = plt.cm.tab20(np.linspace(0, 1, num_sectors))

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        pctdistance=0.75,
        wedgeprops=dict(edgecolor='black', linewidth=0.8)
    )

    # Налаштування стилю тексту
    for autotext in autotexts:
        autotext.set_fontsize(8.5)
        autotext.set_fontweight('bold')
    for text in texts:
        text.set_fontsize(9.5)

    ax.set_title("Кругова діаграма з вийнятими секторами (13 секторів, Варіант 8)", fontsize=14, fontweight='bold', pad=20)
    ax.axis('equal')
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(IMAGES_DIR, "task1_2_pie_chart.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def task1_3_scatter_plot(save=True, show=False):
    """
    1.3. Діаграма розкиду даних:
    - Кількість точок: Варіант + 50 = 8 + 50 = 58 точок.
    - Випадкова генерація даних для осей X та Y.
    - Кожні 10 точок мають різний стиль маркера.
    - Підписи осей, заголовок, легенда.
    """
    num_points = 8 + 50  # 58 точок
    np.random.seed(108)

    x = np.random.uniform(5, 95, size=num_points)
    y = np.random.uniform(5, 95, size=num_points)

    # Стилі маркерів для кожних 10 точок (6 груп для 58 точок)
    marker_styles = [
        ('o', 'Круги (0-9)', '#e74c3c'),
        ('s', 'Квадрати (10-19)', '#3498db'),
        ('^', 'Трикутники (20-29)', '#2ecc71'),
        ('*', 'Зірочки (30-39)', '#f39c12'),
        ('D', 'Ромби (40-49)', '#9b59b6'),
        ('X', 'Хрестики (50-57)', '#1abc9c')
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, (m, label_name, col) in enumerate(marker_styles):
        start = idx * 10
        end = min((idx + 1) * 10, num_points)
        if start < num_points:
            ax.scatter(
                x[start:end],
                y[start:end],
                marker=m,
                color=col,
                s=120,
                alpha=0.85,
                edgecolors='black',
                linewidths=0.7,
                label=f"{label_name}: точки {start + 1}–{end}"
            )

    ax.set_title("Діаграма розкиду даних (58 точок з 6-ма типами маркерів, Варіант 8)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Випадкова величина X (абсциса)", fontsize=12, labelpad=10)
    ax.set_ylabel("Випадкова величина Y (ордината)", fontsize=12, labelpad=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', frameon=True, shadow=True, title="Стилі маркерів")
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(IMAGES_DIR, "task1_3_scatter_plot.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ==============================================================================
# ЗАВДАННЯ 2: ГРАФІЧНІ ПРИМІТИВИ (Варіант 8)
# ==============================================================================

def task2_1_geometric_primitives(save=True, show=False):
    """
    2.1. Додати в графічну область різними кольорами:
    - півколо,
    - стрілку,
    - трикутник,
    - сектор,
    - ромб,
    - прямокутник.
    """
    fig, ax = plt.subplots(figsize=(11, 8))

    # 1. Півколо (Wedge з кутами 0..180)
    semicircle = patches.Wedge(center=(2.5, 7.5), r=1.5, theta1=0, theta2=180,
                               facecolor='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.add_patch(semicircle)
    ax.text(2.5, 7.8, "Півколо", ha='center', va='bottom', fontsize=11, fontweight='bold', color='white')

    # 2. Прямокутник
    rect = patches.Rectangle((6, 6), width=3.5, height=2.2,
                             facecolor='#3498db', edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.add_patch(rect)
    ax.text(7.75, 7.1, "Прямокутник", ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # 3. Трикутник
    triangle_points = np.array([[1.5, 1.5], [4.0, 1.5], [2.75, 4.5]])
    triangle = patches.Polygon(triangle_points, closed=True,
                               facecolor='#2ecc71', edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.add_patch(triangle)
    ax.text(2.75, 2.3, "Трикутник", ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # 4. Сектор (Wedge з кутами 30..120)
    sector = patches.Wedge(center=(8.0, 2.0), r=2.2, theta1=30, theta2=120,
                           facecolor='#f39c12', edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.add_patch(sector)
    ax.text(8.8, 3.2, "Сектор", ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # 5. Ромб
    rhombus_points = np.array([[5.5, 3.5], [6.5, 5.0], [5.5, 6.5], [4.5, 5.0]])
    rhombus = patches.Polygon(rhombus_points, closed=True,
                              facecolor='#9b59b6', edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.add_patch(rhombus)
    ax.text(5.5, 5.0, "Ромб", ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # 6. Стрілка
    arrow = patches.FancyArrow(x=0.5, y=5.5, dx=2.2, dy=0.0,
                               width=0.35, head_width=0.8, head_length=0.7,
                               facecolor='#1abc9c', edgecolor='black', linewidth=1.2)
    ax.add_patch(arrow)
    ax.text(1.3, 6.1, "Стрілка", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#16a085')

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_title("Завдання 2.1: Базові графічні примітиви в Matplotlib", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Координата X", fontsize=11)
    ax.set_ylabel("Координата Y", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(IMAGES_DIR, "task2_1_primitives.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def task2_2_ngon_arrows(save=True, show=False):
    """
    2.2. Кольорова фігура N-багатокутник (N = варіант = 8, октагон),
    контур зробити стрілочками, а в середині розмістити текст.
    """
    N = 8  # Варіант 8 -> восьмикутник
    R = 4.0  # Радіус описаного кола
    center_x, center_y = 0.0, 0.0

    # Вершини правильного восьмикутника
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + (np.pi / N)
    x_coords = center_x + R * np.cos(angles)
    y_coords = center_y + R * np.sin(angles)
    vertices = np.column_stack((x_coords, y_coords))

    fig, ax = plt.subplots(figsize=(9, 9))

    # Заповнений багатокутник
    polygon = patches.Polygon(vertices, closed=True,
                              facecolor='#b8e994', edgecolor='none', alpha=0.65)
    ax.add_patch(polygon)

    # Контур у вигляді спрямованих стрілочок по периметру
    for i in range(N):
        p_start = vertices[i]
        p_end = vertices[(i + 1) % N]
        arrow = patches.FancyArrowPatch(
            posA=p_start,
            posB=p_end,
            arrowstyle='-|>',
            mutation_scale=22,
            color='#0a3d62',
            linewidth=2.8
        )
        ax.add_patch(arrow)

        # Номери вершин для наочності
        v_label_x = p_start[0] * 1.12
        v_label_y = p_start[1] * 1.12
        ax.text(v_label_x, v_label_y, f"$V_{i+1}$", ha='center', va='center',
                fontsize=11, fontweight='bold', color='#1e3799')

    # Текст усередині фігури
    text_content = (
        "N = 8 (Восьмикутник)\n"
        "Варіант 8\n"
        "Контур: стрілочки по периметру\n"
        "R = 4.0, N = 8 вершин"
    )
    ax.text(0, 0, text_content, ha='center', va='center', fontsize=12, fontweight='bold',
            color='#0c2461', bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='#0a3d62'))

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal')
    ax.set_title("Завдання 2.2: N-багатокутник (N = 8) з контуром-стрілочками", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Вісь X", fontsize=11)
    ax.set_ylabel("Вісь Y", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(IMAGES_DIR, "task2_2_octagon.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ==============================================================================
# ЗАВДАННЯ 3: ВАРІАНТ 8
# ==============================================================================

def task3_1_curves(save=True, show=False):
    """
    3.1. Візуалізація кривих:
    а) x^2*y^2 + 2*x*y + x^3 = y^3 (неявна крива)
    б) y = 2*sin^2(t) + sin(2t), x = tg(t) (параметрична крива, еквівалентна y = (2x^2 + 2x)/(x^2 + 1))
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- Графік а): Неявна крива x^2*y^2 + 2*x*y + x^3 - y^3 = 0 ---
    ax1 = axes[0]
    x_vals = np.linspace(-4, 4, 1000)
    y_vals = np.linspace(-4, 4, 1000)
    X, Y = np.meshgrid(x_vals, y_vals)
    F = X ** 2 * Y ** 2 + 2 * X * Y + X ** 3 - Y ** 3

    contour = ax1.contour(X, Y, F, levels=[0], colors='#8e44ad', linewidths=2.5)
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_title(r"а) Неявна крива: $x^2 y^2 + 2xy + x^3 = y^3$", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Вісь X", fontsize=11)
    ax1.set_ylabel("Вісь Y", fontsize=11)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)

    # Ручний елемент для легенди контуру
    proxy = plt.Line2D([0], [0], color='#8e44ad', lw=2.5, label=r'$x^2 y^2 + 2xy + x^3 - y^3 = 0$')
    ax1.legend(handles=[proxy], loc='upper left')

    # --- Графік б): Параметрична крива y = 2*sin^2(t) + sin(2t), x = tg(t) ---
    ax2 = axes[1]
    # Оскільки x = tg(t) має розриви в t = pi/2 та 3pi/2, генеруємо t на інтервалі (-pi/2, pi/2)
    # що охоплює весь діапазон x від -inf до +inf
    t = np.linspace(-np.pi / 2 + 0.05, np.pi / 2 - 0.05, 2000)
    x_param = np.tan(t)
    y_param = 2 * (np.sin(t) ** 2) + np.sin(2 * t)

    ax2.plot(x_param, y_param, color='#2980b9', linewidth=2.5,
             label=r'Параметрично: $x = \mathrm{tg}(t),\ y = 2\sin^2(t) + \sin(2t)$')

    # Горизонтальна асимптота y = 2 при x -> +-inf
    ax2.axhline(2, color='#e74c3c', linestyle=':', linewidth=1.5, label='Асимптота $y = 2$')
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.axvline(0, color='black', linewidth=0.8, linestyle='--')

    # Характерні точки: мінімум (1-sqrt(2), 1-sqrt(2)) та максимум (1+sqrt(2), 1+sqrt(2))
    x_min, y_min = 1 - np.sqrt(2), 1 - np.sqrt(2)
    x_max, y_max = 1 + np.sqrt(2), 1 + np.sqrt(2)
    ax2.plot([x_min, x_max], [y_min, y_max], 'o', color='#c0392b', markersize=6)
    ax2.annotate(f'Min ({x_min:.2f}, {y_min:.2f})', xy=(x_min, y_min), xytext=(x_min - 3.5, y_min - 0.7),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8), fontsize=9)
    ax2.annotate(f'Max ({x_max:.2f}, {y_max:.2f})', xy=(x_max, y_max), xytext=(x_max + 0.5, y_max + 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8), fontsize=9)

    ax2.set_xlim(-10, 10)
    ax2.set_ylim(-1.5, 3.5)
    ax2.set_title(r"б) Параметрична крива ($x = \mathrm{tg}\,t, y = 2\sin^2 t + \sin 2t$)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Вісь X", fontsize=11)
    ax2.set_ylabel("Вісь Y", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', frameon=True)

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(IMAGES_DIR, "task3_1_curves.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def task3_2_polar(save=True, show=False):
    """
    3.2. Крива другого порядку в полярній системі координат:
    rho = a / (1 + b * f(phi)), де a = 5/3, b = -4/3, f(phi) = sin(phi)
    rho = (5/3) / (1 - (4/3) * sin(phi))
    Оскільки ексцентриситет e = 4/3 > 1, це гіпербола.
    Будуємо графік у полярній та декартовій системах для повноти аналізу.
    """
    a = 5.0 / 3.0
    b = -4.0 / 3.0

    fig = plt.figure(figsize=(15, 6))

    # --- 1. Полярна система координат ---
    ax_polar = fig.add_subplot(1, 2, 1, projection='polar')

    # Асимптотичні напрямки: 1 - 4/3 sin(phi) = 0 => sin(phi) = 3/4
    phi_asymp1 = np.arcsin(0.75)
    phi_asymp2 = np.pi - phi_asymp1

    # Будуємо ділянку, де rho > 0 (основна гілка, що охоплює початок координат)
    # phi від phi_asymp2 + delta до phi_asymp1 - delta + 2*pi
    delta = 0.06
    phi_pos = np.linspace(phi_asymp2 + delta, phi_asymp1 - delta + 2 * np.pi, 1000)
    r_pos = a / (1 + b * np.sin(phi_pos))

    ax_polar.plot(phi_pos, r_pos, color='#2980b9', linewidth=2.5, label=r'$\rho = \frac{5/3}{1 - \frac{4}{3}\sin\varphi}$')
    ax_polar.set_title(r"Полярна система: $\rho = \frac{5/3}{1 - \frac{4}{3}\sin\varphi}$", fontsize=13, fontweight='bold', pad=15)
    ax_polar.set_rlim(0, 15)
    ax_polar.grid(True, linestyle=':', alpha=0.7)
    ax_polar.legend(loc='lower right', bbox_to_anchor=(1.15, -0.1))

    # --- 2. Декартова система координат (повна гіпербола з обома гілками та асимптотами) ---
    ax_cart = fig.add_subplot(1, 2, 2)

    # Канонічне рівняння в декартових координатах:
    # (y + 20/7)^2 / (225/49) - x^2 / (25/7) = 1
    # Центр: (0, -20/7) ~ (0, -2.86)
    # y = -20/7 +- (15/7) * sqrt(1 + 7*x^2 / 25)
    x_c = np.linspace(-7, 7, 500)
    y_c_upper = -20.0 / 7.0 + (15.0 / 7.0) * np.sqrt(1.0 + (7.0 * x_c ** 2) / 25.0)
    y_c_lower = -20.0 / 7.0 - (15.0 / 7.0) * np.sqrt(1.0 + (7.0 * x_c ** 2) / 25.0)

    ax_cart.plot(x_c, y_c_upper, color='#2980b9', linewidth=2.2, label='Гіпербола (верхня гілка, фокус (0,0))')
    ax_cart.plot(x_c, y_c_lower, color='#e67e22', linewidth=2.2, label='Гіпербола (нижня гілка)')

    # Асимптоти гіперболи: y + 20/7 = +- (15 / (5*sqrt(7))) x = +- (3/sqrt(7)) x
    slope = 3.0 / np.sqrt(7.0)
    y_asymp_up = -20.0 / 7.0 + slope * x_c
    y_asymp_down = -20.0 / 7.0 - slope * x_c
    ax_cart.plot(x_c, y_asymp_up, 'k--', linewidth=1.2, alpha=0.7, label='Асимптоти')
    ax_cart.plot(x_c, y_asymp_down, 'k--', linewidth=1.2, alpha=0.7)

    # Фокус на початку координат
    ax_cart.plot(0, 0, 'ro', markersize=7, label='Фокус $F_1(0, 0)$')
    # Центр
    ax_cart.plot(0, -20 / 7, 'bs', markersize=6, label=r'Центр $(0, -2.86)$')

    ax_cart.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax_cart.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax_cart.set_xlim(-6, 6)
    ax_cart.set_ylim(-9, 4)
    ax_cart.set_title("Декартова система: канонічний вигляд гіперболи", fontsize=13, fontweight='bold')
    ax_cart.set_xlabel("Вісь X", fontsize=11)
    ax_cart.set_ylabel("Вісь Y", fontsize=11)
    ax_cart.grid(True, linestyle=':', alpha=0.6)
    ax_cart.legend(loc='lower right', fontsize=8.5)

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(IMAGES_DIR, "task3_2_polar.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def task3_3_color_hyperbola(save=True, show=False):
    """
    3.3. Побудувати кольорову криву, використовуючи принаймні три різні кольори.
    Додати підпис "Вказати рівняння кривої" під кутом alpha:
    xy + 2x + y + 5/2 = 0, alpha = 58 градусів.
    
    Аналітичний вигляд:
    (x + 1)(y + 2) = -0.5  =>  y = -2 - 1 / (2*(x + 1))
    Рівнобічна гіпербола з асимптотами x = -1 та y = -2.
    Для вимоги "принаймні три різні кольори" розбиваємо гілки на різнокольорові сегменти
    та використовуємо градієнт/спектральну палітру.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Гілка 1: x < -1
    x1_a = np.linspace(-6.0, -2.5, 300)
    y1_a = -2.0 - 0.5 / (x1_a + 1.0)

    x1_b = np.linspace(-2.5, -1.08, 300)
    y1_b = -2.0 - 0.5 / (x1_b + 1.0)

    # Гілка 2: x > -1
    x2_a = np.linspace(-0.92, 0.5, 300)
    y2_a = -2.0 - 0.5 / (x2_a + 1.0)

    x2_b = np.linspace(0.5, 4.0, 300)
    y2_b = -2.0 - 0.5 / (x2_b + 1.0)

    # Побудова кривої за допомогою 4-х різних кольорів (>= 3 кольори за умовою)
    ax.plot(x1_a, y1_a, color='#2980b9', linewidth=3.0, label='Сегмент 1 (Синій): $x \\in [-6, -2.5]$')
    ax.plot(x1_b, y1_b, color='#27ae60', linewidth=3.0, label='Сегмент 2 (Зелений): $x \\in (-2.5, -1)$')
    ax.plot(x2_a, y2_a, color='#e67e22', linewidth=3.0, label='Сегмент 3 (Помаранчевий): $x \\in (-1, 0.5)$')
    ax.plot(x2_b, y2_b, color='#9b59b6', linewidth=3.0, label='Сегмент 4 (Фіолетовий): $x \\in [0.5, 4]$')

    # Асимптоти
    ax.axvline(-1, color='#e74c3c', linestyle='--', linewidth=1.5, label='Асимптота $x = -1$')
    ax.axhline(-2, color='#e74c3c', linestyle='--', linewidth=1.5, label='Асимптота $y = -2$')

    # Центр симетрії гіперболи (-1, -2)
    ax.plot(-1, -2, 'ko', markersize=6, label='Центр $(-1, -2)$')

    # Осі координат
    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax.axvline(0, color='black', linewidth=0.8, linestyle=':')

    # Додавання підпису під кутом alpha = 58 градусів (за методичкою)
    alpha = 58
    label_text = r"$xy + 2x + y + \frac{5}{2} = 0$"
    ax.text(0.0, -0.6, label_text, fontsize=12, fontweight='bold', color='#c0392b',
            rotation=alpha, rotation_mode='anchor', ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fcf3cf', edgecolor='#c0392b', alpha=0.9))

    ax.set_xlim(-6, 4)
    ax.set_ylim(-7, 3)
    ax.set_title(r"Завдання 3.3: Кольорова гіпербола з підписом під кутом $\alpha = 58^\circ$",
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Вісь X", fontsize=12)
    ax.set_ylabel("Вісь Y", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', frameon=True, shadow=True, fontsize=9.5)
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(IMAGES_DIR, "task3_3_color_hyperbola.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def task3_4_functions_and_intersection(save=True, show=False):
    """
    3.4. Графіки функцій f(x) = x^3 - 1 та g(x) = sin^2(x + 1):
    1) Два підграфіки в одному вікні (коричневий та зелений).
    2) Обидва графіки у декартовій системі з:
       - чітко виділеними точками перетину;
       - оформленими осями ("Вісь X", "Вісь Y");
       - підписами графіків (математичні вирази);
       - суцільними лініями різної товщини.
    """
    # Визначення математичних функцій
    def f(x):
        return x ** 3 - 1.0

    def g(x):
        return np.sin(x + 1.0) ** 2

    # Знаходження точки перетину чисельно: f(x) - g(x) = 0
    # На проміжку [1.0, 1.3] f(x) монотонно зростає від 0 до 1.197, g(x) знаходиться в [0, 1]
    x_root = brentq(lambda x: f(x) - g(x), 1.0, 1.3)
    y_root = f(x_root)

    print(f"[Task 3.4] Точка перетину знайдена: x = {x_root:.6f}, y = {y_root:.6f}")
    print(f"            f(x*) = {f(x_root):.6f}, g(x*) = {g(x_root):.6f}")

    x_vals = np.linspace(-2.0, 2.0, 1000)
    y_f = f(x_vals)
    y_g = g(x_vals)

    # --------------------------------------------------------------------------
    # Частина 1: Два підграфіки в одному вікні
    # --------------------------------------------------------------------------
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Підграфік 1: f(x) = x^3 - 1 (коричневий)
    ax1.plot(x_vals, y_f, color='saddlebrown', linewidth=2.5, label=r'$f(x) = x^3 - 1$')
    ax1.axhline(0, color='black', linewidth=1.0, linestyle='--')
    ax1.axvline(0, color='black', linewidth=1.0, linestyle='--')
    ax1.set_xlim(-2.0, 2.0)
    ax1.set_ylim(-8.0, 8.0)
    ax1.set_title(r"Графік функції $f(x) = x^3 - 1$", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Вісь X", fontsize=11)
    ax1.set_ylabel("Вісь Y", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=11)

    # Підграфік 2: g(x) = sin^2(x + 1) (зелений)
    ax2.plot(x_vals, y_g, color='forestgreen', linewidth=2.5, label=r'$g(x) = \sin^2(x + 1)$')
    ax2.axhline(0, color='black', linewidth=1.0, linestyle='--')
    ax2.axvline(0, color='black', linewidth=1.0, linestyle='--')
    ax2.set_xlim(-2.0, 2.0)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_title(r"Графік функції $g(x) = \sin^2(x + 1)$", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Вісь X", fontsize=11)
    ax2.set_ylabel("Вісь Y", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=11)

    fig1.suptitle("Завдання 3.4 (Частина 1): Окремі підграфіки функцій f(x) та g(x)",
                  fontsize=15, fontweight='bold', y=0.98)
    fig1.tight_layout()

    if save:
        fig1.savefig(os.path.join(IMAGES_DIR, "task3_4_subplots.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig1)

    # --------------------------------------------------------------------------
    # Частина 2: Обидва графіки на одній площині з точкою перетину
    # --------------------------------------------------------------------------
    fig2, ax = plt.subplots(figsize=(11, 7))

    # Обидва графіки суцільною лінією різної товщини:
    # f(x) - товщина 3.2, g(x) - товщина 1.8
    ax.plot(x_vals, y_f, color='saddlebrown', linestyle='-', linewidth=3.2,
            label=r'$f(x) = x^3 - 1$ (товщина 3.2, коричнева)')
    ax.plot(x_vals, y_g, color='forestgreen', linestyle='-', linewidth=1.8,
            label=r'$g(x) = \sin^2(x + 1)$ (товщина 1.8, зелена)')

    # Оформлення осей координат
    ax.axhline(0, color='black', linewidth=1.4, linestyle='-')
    ax.axvline(0, color='black', linewidth=1.4, linestyle='-')

    # Позначення точки перетину
    ax.plot(x_root, y_root, 'ro', markersize=9, markeredgecolor='black', markeredgewidth=1.2,
            label=f'Точка перетину: ({x_root:.2f}, {y_root:.2f})')

    # Анотація зі стрілкою
    ax.annotate(
        f'Перетин: ({x_root:.3f}, {y_root:.3f})\n$f(x^*) = g(x^*) \\approx {y_root:.3f}$',
        xy=(x_root, y_root),
        xytext=(x_root - 1.2, y_root + 1.8),
        fontsize=11,
        fontweight='bold',
        color='#900C3F',
        arrowprops=dict(facecolor='#900C3F', edgecolor='black', arrowstyle='->', lw=1.5, shrinkA=0, shrinkB=5),
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9ebea', edgecolor='#c0392b', alpha=0.95)
    )

    # Підписи осей відповідно до вимог методички
    ax.set_xlabel("Вісь X", fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel("Вісь Y", fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title("Завдання 3.4 (Частина 2): Спільна система координат та точка перетину",
                 fontsize=14, fontweight='bold', pad=15)

    ax.set_xlim(-1.8, 2.0)
    ax.set_ylim(-3.5, 4.0)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=10.5)

    plt.tight_layout()

    if save:
        fig2.savefig(os.path.join(IMAGES_DIR, "task3_4_combined.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig2)


# ==============================================================================
# ТОЧКА ВХОДУ ДЛЯ ЗАПУСКУ ВСІХ ЗАВДАНЬ
# ==============================================================================

def main():
    print("=" * 60)
    print("  ВИКОНАННЯ ЛАБОРАТОРНОЇ РОБОТИ № 2 З ПРИКЛАДНОЇ МАТЕМАТИКИ")
    print("  Виконав: Дацишин Володимир, група ІПЗ-23, Варіант: 8")
    print("=" * 60)

    print("\n--- Завдання 1.1: Стовпчаста діаграма (13 груп, 4 сегменти) ---")
    task1_1_stacked_bar(save=True, show=False)
    print("  [OK] Збережено: images/task1_1_stacked_bar.png")

    print("\n--- Завдання 1.2: Кругова діаграма (13 секторів з виділенням) ---")
    task1_2_pie_chart(save=True, show=False)
    print("  [OK] Збережено: images/task1_2_pie_chart.png")

    print("\n--- Завдання 1.3: Діаграма розкиду (58 точок з 6 типами маркерів) ---")
    task1_3_scatter_plot(save=True, show=False)
    print("  [OK] Збережено: images/task1_3_scatter_plot.png")

    print("\n--- Завдання 2.1: Базові графічні примітиви ---")
    task2_1_geometric_primitives(save=True, show=False)
    print("  [OK] Збережено: images/task2_1_primitives.png")

    print("\n--- Завдання 2.2: 8-кутник з контуром-стрілочками та текстом ---")
    task2_2_ngon_arrows(save=True, show=False)
    print("  [OK] Збережено: images/task2_2_octagon.png")

    print("\n--- Завдання 3.1: Візуалізація кривих а) та б) ---")
    task3_1_curves(save=True, show=False)
    print("  [OK] Збережено: images/task3_1_curves.png")

    print("\n--- Завдання 3.2: Крива другого порядку в полярній системі ---")
    task3_2_polar(save=True, show=False)
    print("  [OK] Збережено: images/task3_2_polar.png")

    print("\n--- Завдання 3.3: Кольорова гіпербола з підписом під кутом 58° ---")
    task3_3_color_hyperbola(save=True, show=False)
    print("  [OK] Збережено: images/task3_3_color_hyperbola.png")

    print("\n--- Завдання 3.4: Графіки f(x) та g(x), точка перетину ---")
    task3_4_functions_and_intersection(save=True, show=False)
    print("  [OK] Збережено: images/task3_4_subplots.png")
    print("  [OK] Збережено: images/task3_4_combined.png")

    print("\n" + "=" * 60)
    print("  ВСІ ЗАВДАННЯ ЛАБОРАТОРНОЇ РОБОТИ № 2 УСПІШНО ВИКОНАНО!")
    print(f"  Всі графіки збережено у папку: {IMAGES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()