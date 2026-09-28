"""
Лабораторна робота №1: Комп'ютерна графіка та візуалізація.
Обробка зображень: формати, зміна розміру, заміна кольорів, баланс та спектр.
"""
from __future__ import annotations
import os
from PIL import Image
import numpy as np


def get_size_kb(filepath: str) -> float:
    """Повертає розмір файлу в кілобайтах."""
    return os.path.getsize(filepath) / 1024.0 if os.path.isfile(filepath) else 0.0


def parse_color(color_str: str) -> tuple[int, int, int]:
    """Розпізнає колір за назвою, HEX або десятковим форматом 'R,G,B'."""
    s = color_str.strip().lower()
    palette = {
        "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
        "white": (255, 255, 255), "black": (0, 0, 0), "yellow": (255, 255, 0),
        "cyan": (0, 255, 255), "magenta": (255, 0, 255), "gray": (128, 128, 128)
    }
    if s in palette:
        return palette[s]
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 6 and all(c in "0123456789abcdef" for c in s):
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    parts = [int(p.strip()) for p in color_str.split(",") if p.strip()]
    if len(parts) >= 3 and all(0 <= p <= 255 for p in parts[:3]):
        return (parts[0], parts[1], parts[2])
    raise ValueError(f"Невірний формат кольору: '{color_str}'")


# 1. Конвертація форматів зображень
def convert_formats(file_paths: list[str], formats: list[str], output_dir: str) -> None:
    """Конвертує зображення у задані формати та порівнює розміри."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'Файл':<20} | {'Формат':<7} | {'До (KB)':<10} | {'Після (KB)':<10} | Різниця")
    print("-" * 65)

    for path in file_paths:
        if not os.path.isfile(path):
            continue
        try:
            with Image.open(path) as img:
                base_name = os.path.splitext(os.path.basename(path))[0]
                size_before = get_size_kb(path)
                for fmt in formats:
                    fmt = fmt.strip().lstrip(".").lower()
                    save_img = img.copy()
                    if fmt in ("jpg", "jpeg", "bmp") and save_img.mode in ("RGBA", "LA", "P"):
                        save_img = save_img.convert("RGBA")
                        bg = Image.new("RGB", save_img.size, (255, 255, 255))
                        bg.paste(save_img, mask=save_img.split()[3])
                        save_img = bg
                    elif fmt in ("jpg", "jpeg") and save_img.mode != "RGB":
                        save_img = save_img.convert("RGB")

                    out_path = os.path.join(output_dir, f"{base_name}.{fmt}")
                    save_img.save(out_path)
                    size_after = get_size_kb(out_path)
                    diff = size_after - size_before
                    print(f"{base_name[:20]:<20} | {fmt.upper():<7} | {size_before:<10.2f} | {size_after:<10.2f} | {diff:+.2f} KB")
        except Exception as e:
            print(f"Помилка конвертації '{path}': {e}")
    print("-" * 65)


# 2. Зміна розміру зображень
RESAMPLE_METHODS = {
    "1": (Image.Resampling.LANCZOS, "Lanczos"),
    "2": (Image.Resampling.BICUBIC, "Bicubic"),
    "3": (Image.Resampling.BILINEAR, "Bilinear"),
    "4": (Image.Resampling.NEAREST, "Nearest"),
}


def resize_images(
    file_paths: list[str], output_dir: str, target_w: int | None = None, target_h: int | None = None, filter_choice: str = "1"
) -> None:
    """Зміна розміру із вибором інтерполяції та збереженням пропорцій."""
    os.makedirs(output_dir, exist_ok=True)
    resample_filter, filter_name = RESAMPLE_METHODS.get(filter_choice, (Image.Resampling.LANCZOS, "Lanczos"))
    print(f"\nЗастосовано фільтр: {filter_name}")
    print(f"{'Файл':<20} | {'Було':<11} | {'Стало':<11} | {'До (KB)':<10} | Після (KB)")
    print("-" * 65)

    for path in file_paths:
        if not os.path.isfile(path):
            continue
        try:
            with Image.open(path) as img:
                w, h = img.size
                if target_w and target_h:
                    nw, nh = target_w, target_h
                elif target_w:
                    nw, nh = target_w, int(round(h * target_w / w))
                elif target_h:
                    nw, nh = int(round(w * target_h / h)), target_h
                else:
                    return

                resized = img.resize((nw, nh), resample=resample_filter)
                base_name, ext = os.path.splitext(os.path.basename(path))
                out_path = os.path.join(output_dir, f"{base_name}_{nw}x{nh}{ext}")
                resized.save(out_path)
                print(f"{base_name[:20]:<20} | {f'{w}x{h}':<11} | {f'{nw}x{nh}':<11} | {get_size_kb(path):<10.2f} | {get_size_kb(out_path):.2f}")
        except Exception as e:
            print(f"Помилка зміни розміру '{path}': {e}")
    print("-" * 65)


# 3. Перетворення кольорів
def replace_color(
    input_path: str, output_path: str, target_rgb: tuple[int, int, int], new_rgb: tuple[int, int, int], tolerance: int = 0
) -> bool:
    """Заміна цільового кольору на новий з урахуванням похибки tolerance."""
    if not os.path.isfile(input_path):
        return False
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with Image.open(input_path) as img:
        work_img = img.convert("RGBA" if img.mode == "RGBA" else "RGB")
        arr = np.array(work_img)
        diff = np.abs(arr[:, :, :3].astype(np.int32) - np.array(target_rgb, dtype=np.int32))
        mask = np.all(diff <= tolerance, axis=-1)
        replaced = int(np.count_nonzero(mask))
        arr[mask, :3] = new_rgb

        result = Image.fromarray(arr)
        if output_path.lower().endswith((".jpg", ".jpeg")) and result.mode != "RGB":
            result = result.convert("RGB")
        result.save(output_path)

        total = img.width * img.height
        pct = (replaced / total * 100) if total > 0 else 0.0
        print(f"Збережено: {output_path} | Замінено: {replaced:,} з {total:,} пікселів ({pct:.2f}%)")
        return True


# 4. Корекція колірного балансу та аналіз спектру
def adjust_color_balance(
    input_path: str, output_path: str, r_shift: int = 0, g_shift: int = 0, b_shift: int = 0
) -> bool:
    """Коригує яскравість каналів R, G, B у діапазоні [-255..255]."""
    if not os.path.isfile(input_path):
        return False
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with Image.open(input_path) as img:
        arr = np.array(img.convert("RGB"), dtype=np.int16)
        arr[:, :, 0] += r_shift
        arr[:, :, 1] += g_shift
        arr[:, :, 2] += b_shift
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(output_path)
        print(f"Збережено: {output_path} (зсув R={r_shift:+d}, G={g_shift:+d}, B={b_shift:+d})")
        return True


def auto_color_balance(input_path: str, output_path: str, target: int = 128) -> bool:
    """Автоматичний баланс каналів за методичкою (вирівнювання до рівня 128)."""
    with Image.open(input_path) as img:
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        r_shift = int(round(target - float(np.mean(arr[:, :, 0]))))
        g_shift = int(round(target - float(np.mean(arr[:, :, 1]))))
        b_shift = int(round(target - float(np.mean(arr[:, :, 2]))))
        print(f"Зсуви каналів до {target}: R={r_shift:+d}, G={g_shift:+d}, B={b_shift:+d}")
        return adjust_color_balance(input_path, output_path, r_shift, g_shift, b_shift)


def analyze_spectrum(input_path: str) -> None:
    """Аналіз спектру зображення та виявлення домінування каналів (стор. 30 методички)."""
    with Image.open(input_path) as img:
        arr = np.array(img.convert("RGB"), dtype=np.float64)
        means = [float(np.mean(arr[:, :, i])) for i in range(3)]
        total_energy = sum(means) or 1.0
        pcts = [(m / total_energy) * 100 for m in means]
        print(f"\nАналіз спектру: {os.path.basename(input_path)} ({img.width}x{img.height})")
        print(f"  • Червоний (R): {means[0]:>5.1f} / 255  ({pcts[0]:>4.1f}%)")
        print(f"  • Зелений  (G): {means[1]:>5.1f} / 255  ({pcts[1]:>4.1f}%)")
        print(f"  • Синій    (B): {means[2]:>5.1f} / 255  ({pcts[2]:>4.1f}%)")
        if max(pcts) - min(pcts) < 5.0:
            print("  ✓ Спектр збалансований (частки близькі до 33.3%).")
        else:
            names = ["червоний", "зелений", "синій"]
            dom = [names[i] for i in range(3) if pcts[i] > 38.0]
            print(f"  ! Домінування: {', '.join(dom) if dom else 'дисбаланс часток'}.")


def extract_channels(input_path: str, output_dir: str) -> None:
    """Виділяє та зберігає канали RGB окремо (стор. 29-30 методички)."""
    os.makedirs(output_dir, exist_ok=True)
    with Image.open(input_path) as img:
        arr = np.array(img.convert("RGB"))
        base = os.path.splitext(os.path.basename(input_path))[0]
        print("\nЗбереження окремих каналів:")
        for idx, name in enumerate(("red", "green", "blue")):
            ch = np.zeros_like(arr)
            ch[:, :, idx] = arr[:, :, idx]
            out_path = os.path.join(output_dir, f"{base}_{name}.png")
            Image.fromarray(ch).save(out_path)
            print(f"  • Канал {name.capitalize():<5}: {out_path}")


# 5. Головне меню
def main() -> None:
    """Консольний інтерфейс користувача для виконання лабораторної роботи."""
    while True:
        print("\n=== Лабораторна робота №1: Комп'ютерна графіка ===")
        print("1. Конвертація форматів (з порівнянням розмірів)")
        print("2. Зміна розміру зображень (пропорційна, точна)")
        print("3. Перетворення кольорів (заміна цільового кольору)")
        print("4. Корекція колірного балансу та спектр\n5. Вихід")

        try:
            choice = input("Оберіть дію (1-5): ").strip().strip('"\'')
        except (EOFError, KeyboardInterrupt):
            print("\nРоботу завершено.")
            break

        if choice == "1":
            paths = [p.strip().strip('"\'') for p in input("Шляхи через кому: ").split(",") if p.strip()]
            fmts = [f.strip() for f in input("Формати (png, jpg, webp, bmp): ").split(",") if f.strip()]
            out_dir = input("Директорія збереження: ").strip().strip('"\'')
            if paths and fmts and out_dir:
                convert_formats(paths, fmts, out_dir)

        elif choice == "2":
            paths = [p.strip().strip('"\'') for p in input("Шляхи через кому: ").split(",") if p.strip()]
            out_dir = input("Директорія збереження: ").strip().strip('"\'')
            if not paths or not out_dir:
                continue
            mode = input("Режим (a. Точні WxH / b. Лише W / c. Лише H): ").strip().lower()
            try:
                target_w = int(input("Ширина: ")) if mode in ("a", "b") else None
                target_h = int(input("Висота: ")) if mode in ("a", "c") else None
                filt = input("Фільтр (1.Lanczos 2.Bicubic 3.Bilinear 4.Nearest, дефолт 1): ").strip() or "1"
                resize_images(paths, out_dir, target_w, target_h, filt)
            except ValueError as e:
                print(f"Помилка вводу: {e}")

        elif choice == "3":
            try:
                in_path = input("Вхідний файл: ").strip().strip('"\'')
                out_path = input("Вихідний файл: ").strip().strip('"\'')
                c_from = parse_color(input("Колір для заміни (назва, HEX або R,G,B): "))
                c_to = parse_color(input("Новий колір: "))
                tol = int(input("Похибка (0-255, дефолт 0): ").strip() or "0")
                replace_color(in_path, out_path, c_from, c_to, tol)
            except Exception as e:
                print(f"Помилка: {e}")

        elif choice == "4":
            in_path = input("Шлях до зображення: ").strip().strip('"\'')
            if not os.path.isfile(in_path):
                print(f"Файл не знайдено: {in_path}")
                continue

            print("1. Зсув яскравості  2. Баланс R,G,B  3. Авто-баланс  4. Спектр  5. Канали")
            sub = input("Оберіть підпункт (1-5): ").strip()
            try:
                if sub in ("1", "2", "3"):
                    out_path = input("Шлях збереження: ").strip().strip('"\'')
                    if sub == "1":
                        val = int(input("Зсув яскравості [-255..255]: ") or "0")
                        adjust_color_balance(in_path, out_path, val, val, val)
                    elif sub == "2":
                        r_val = int(input("Зсув R [-255..255]: ") or "0")
                        g_val = int(input("Зсув G [-255..255]: ") or "0")
                        b_val = int(input("Зсув B [-255..255]: ") or "0")
                        adjust_color_balance(in_path, out_path, r_val, g_val, b_val)
                    elif sub == "3":
                        target_lvl = int(input("Цільовий рівень (дефолт 128): ").strip() or "128")
                        auto_color_balance(in_path, out_path, target_lvl)
                elif sub == "4":
                    analyze_spectrum(in_path)
                elif sub == "5":
                    extract_channels(in_path, input("Директорія для каналів: ").strip().strip('"\''))
            except Exception as e:
                print(f"Помилка: {e}")

        elif choice == "5":
            print("Роботу завершено. До побачення!")
            break


if __name__ == "__main__":
    main()
