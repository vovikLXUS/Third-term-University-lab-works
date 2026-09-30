"""
Laboratory Work #2: Computer Graphics and Visualization.
Topic: Adding transparency, cropping, and contrast adjustment of images.
Author: Volodymyr Datsyshyn, Group IPZ-23.
"""
from __future__ import annotations
import os
import sys
import numpy as np
from PIL import Image


# HELPER FUNCTIONS
def save_image(img: Image.Image, output_path: str) -> None:
    """Saves an image, creating parent directories and converting RGBA to RGB for JPEG."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()
    if ext in (".jpg", ".jpeg") and img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    img.save(output_path)
    size_kb = os.path.getsize(output_path) / 1024.0
    print(f"Saved: {output_path} ({img.width}x{img.height}, {size_kb:.2f} KB)")


# TASK 1: IMAGE TRANSPARENCY AND LAYERS
def set_transparency(input_path: str, output_path: str, alpha: int = 128) -> None:
    """
    Modifies image transparency using the alpha channel.
    alpha: value from 0 (fully transparent) to 255 (fully opaque).
    """
    with Image.open(input_path) as img:
        rgba = img.convert("RGBA")
        rgba.putalpha(int(np.clip(alpha, 0, 255)))
        save_image(rgba, output_path)
        print(f"  • Alpha channel set to: {alpha}/255 ({alpha / 255 * 100:.1f}%)")


def make_color_transparent(input_path: str, output_path: str, target_rgb=(240, 240, 240), tolerance: int = 15) -> None:
    """Removes a solid background color (Chroma Key), converting it to transparent."""
    with Image.open(input_path) as img:
        arr = np.array(img.convert("RGBA"), dtype=np.int32)
        diff = np.abs(arr[:, :, :3] - np.array(target_rgb, dtype=np.int32))
        mask = np.all(diff <= tolerance, axis=-1)
        arr[mask, 3] = 0  # Zero out alpha channel for background pixels
        res = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
        save_image(res, output_path)
        print(f"  • Made {np.count_nonzero(mask):,} pixels of color RGB{target_rgb} transparent")


def overlay_layer(base_path: str, overlay_path: str, output_path: str, pos=(0, 0), opacity: float = 1.0) -> None:
    """
    Overlays a semi-transparent layer onto a base image with adjustable opacity and position.
    """
    with Image.open(base_path) as base, Image.open(overlay_path) as over:
        base = base.convert("RGBA")
        over = over.convert("RGBA")
        if opacity < 1.0:
            arr = np.array(over, dtype=np.float32)
            arr[:, :, 3] = np.clip(arr[:, :, 3] * opacity, 0, 255)
            over = Image.fromarray(arr.astype(np.uint8), mode="RGBA")

        canvas = Image.new("RGBA", base.size, (0, 0, 0, 0))
        canvas.paste(over, pos)
        result = Image.alpha_composite(base, canvas)
        save_image(result, output_path)
        print(f"  • Overlaid {over.size} layer at position {pos} with opacity {opacity * 100:.0f}%")


# TASK 2: CROPPING AND SPLITTING IMAGES
def split_image(input_path: str, output_dir: str, num_parts: int = 3, orientation: str = "v") -> list[str]:
    """
    Splits an image into a chosen number of equal segments.
    orientation: 'v' - vertical strips (rule of thirds), 'h' - horizontal strips.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated = []
    with Image.open(input_path) as img:
        w, h = img.size
        ext = os.path.splitext(input_path)[1] or ".png"
        for i in range(num_parts):
            if orientation == "v":
                left = int(round(i * w / num_parts))
                right = int(round((i + 1) * w / num_parts))
                box = (left, 0, right, h)
            else:
                top = int(round(i * h / num_parts))
                bottom = int(round((i + 1) * h / num_parts))
                box = (0, top, w, bottom)

            part = img.crop(box)
            out_file = os.path.join(output_dir, f"third_r1_c{i+1}_{i+1}{ext}")
            save_image(part, out_file)
            generated.append(out_file)
        print(f"  • Image {w}x{h} split into {num_parts} parts in '{output_dir}'")
    return generated


def crop_keep_region(input_path: str, output_path: str, box: tuple[int, int, int, int]) -> None:
    """Crops the image to keep only the selected bounding box, discarding outside areas."""
    with Image.open(input_path) as img:
        x1, y1, x2, y2 = box
        cropped = img.crop((x1, y1, x2, y2))
        save_image(cropped, output_path)
        print(f"  • Kept region {box}: new dimensions {cropped.width}x{cropped.height}")


def crop_remove_region(input_path: str, output_path: str, box: tuple[int, int, int, int], fill_color=None) -> None:
    """
    Removes the selected bounding box: makes it transparent or fills with solid color.
    """
    with Image.open(input_path) as img:
        x1, y1, x2, y2 = box
        if fill_color is None:  # Transparent cutout (Alpha = 0)
            res = img.convert("RGBA")
            arr = np.array(res)
            arr[y1:y2, x1:x2, 3] = 0
            res = Image.fromarray(arr, mode="RGBA")
        else:  # Solid color fill
            res = img.convert("RGB")
            arr = np.array(res)
            arr[y1:y2, x1:x2, :3] = fill_color
            res = Image.fromarray(arr, mode="RGB")
        save_image(res, output_path)
        print(f"  • Removed rectangular region {box} ({x2 - x1}x{y2 - y1} px)")


# TASK 3: CONTRAST ADJUSTMENT
def adjust_contrast(input_path: str, output_path: str, k: float = 1.5) -> None:
    """
    Adjusts image contrast using the formula:
        NewY := K * (OldY - AveY) + AveY
    where AveY is the mean channel brightness, K is the contrast factor.
    """
    with Image.open(input_path) as img:
        has_alpha = (img.mode == "RGBA")
        work_img = img.convert("RGBA" if has_alpha else "RGB")
        arr = np.array(work_img, dtype=np.float32)

        mean_before = float(np.mean(arr[:, :, :3]))
        std_before = float(np.std(arr[:, :, :3]))

        # Two-pass algorithm: compute AveY per channel and clip NewY to [0..255]
        for c in range(3):
            ave_y = float(np.mean(arr[:, :, c]))
            arr[:, :, c] = np.clip(k * (arr[:, :, c] - ave_y) + ave_y, 0, 255)

        mean_after = float(np.mean(arr[:, :, :3]))
        std_after = float(np.std(arr[:, :, :3]))

        mode = "RGBA" if has_alpha else "RGB"
        result = Image.fromarray(arr.astype(np.uint8), mode=mode)
        save_image(result, output_path)
        print(f"  • Factor K = {k:.2f} | Contrast σ: {std_before:.1f} -> {std_after:.1f} "
              f"({(std_after - std_before) / std_before * 100:+.1f}%) | AveY: {mean_before:.1f} -> {mean_after:.1f}")


# INTERACTIVE CLI MENU
def main() -> None:
    while True:
        print("\n" + "=" * 55)
        print("   LABORATORY WORK #2: IMAGE PROCESSING")
        print("=" * 55)
        print("1. Change transparency (putalpha)")
        print("2. Remove background color (Chroma Key)")
        print("3. Overlay semi-transparent layer")
        print("4. Split image into N parts")
        print("5. Crop: keep selected region")
        print("6. Crop: remove selected region (transparent/color)")
        print("7. Adjust contrast (NewY formula)")
        print("8. Exit")
        print("-" * 55)

        try:
            choice = input("Select an option (1-8): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting program.")
            break

        if choice == "1":
            inp = input("Enter input image path: ").strip()
            out = input("Enter output image path: ").strip()
            val = int(input("Enter alpha value (0-255): ").strip())
            set_transparency(inp, out, val)

        elif choice == "2":
            inp = input("Enter input image path: ").strip()
            out = input("Enter output image path: ").strip()
            make_color_transparent(inp, out)

        elif choice == "3":
            base = input("Enter base image path: ").strip()
            over = input("Enter overlay image path: ").strip()
            out = input("Enter output image path: ").strip()
            pos_x = int(input("Enter X coordinate for overlay: ").strip())
            pos_y = int(input("Enter Y coordinate for overlay: ").strip())
            opac = float(input("Enter overlay opacity (0.0..1.0): ").strip())
            overlay_layer(base, over, out, pos=(pos_x, pos_y), opacity=opac)

        elif choice == "4":
            inp = input("Enter input image path: ").strip()
            out_dir = input("Enter output directory for parts: ").strip()
            n = int(input("Enter number of parts: ").strip())
            split_image(inp, out_dir, num_parts=n)

        elif choice == "5":
            inp = input("Enter input image path: ").strip()
            out = input("Enter output image path: ").strip()
            box_str = input("Enter coordinates X1,Y1,X2,Y2 (comma-separated): ").strip()
            box = tuple(map(int, box_str.split(",")))
            crop_keep_region(inp, out, box)

        elif choice == "6":
            inp = input("Enter input image path: ").strip()
            out = input("Enter output image path: ").strip()
            box_str = input("Enter coordinates X1,Y1,X2,Y2 (comma-separated): ").strip()
            mode = input("Select removal mode (1 - transparent window, 2 - white fill): ").strip()
            box = tuple(map(int, box_str.split(",")))
            crop_remove_region(inp, out, box, fill_color=(255, 255, 255) if mode == "2" else None)

        elif choice == "7":
            inp = input("Enter input image path: ").strip()
            out = input("Enter output image path: ").strip()
            k = float(input("Enter contrast factor K (>1 increase, <1 decrease): ").strip())
            adjust_contrast(inp, out, k)

        elif choice == "8":
            print("\nProgram finished!")
            break


if __name__ == "__main__":
    main()
