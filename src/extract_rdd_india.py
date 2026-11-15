"""
Extracts and converts the India road damage subset from RDD_SPLIT in archive (2).zip
Converts Class 4 (Pothole/D40) -> Class 0 (pothole), preserving negative samples.
"""

from pathlib import Path
import time
import zipfile
import yaml


def extract_and_prepare_india_subset(
    zip_path: str = r"C:\Users\Asus\Downloads\archive (2).zip",
    target_dir: str = r"C:\Users\Asus\.gemini\antigravity\scratch\roadwatch-ai\data\processed",
):
    start_time = time.time()
    zip_file = Path(zip_path)
    dst_root = Path(target_dir)

    print(f"Opening archive: {zip_file}")
    if not zip_file.is_file():
        raise FileNotFoundError(f"Archive not found: {zip_file}")

    # Prepare directories
    for split in ["train", "val", "test"]:
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {
        "train": {"total_images": 0, "pothole_images": 0, "pothole_boxes": 0, "negative_images": 0},
        "val": {"total_images": 0, "pothole_images": 0, "pothole_boxes": 0, "negative_images": 0},
        "test": {"total_images": 0, "pothole_images": 0, "pothole_boxes": 0, "negative_images": 0},
    }

    with zipfile.ZipFile(zip_file, "r") as z:
        namelist = z.namelist()
        print(f"Total files in archive: {len(namelist)}")

        # Collect India files
        # RDD_SPLIT/{split}/images/India_...jpg
        # RDD_SPLIT/{split}/labels/India_...txt
        india_images = [n for n in namelist if "india" in n.lower() and n.endswith(".jpg")]
        print(f"Total India images found: {len(india_images)}")

        for idx, img_entry in enumerate(india_images, 1):
            parts = img_entry.split("/")
            # parts: ['RDD_SPLIT', 'train', 'images', 'India_000001.jpg']
            if len(parts) < 4:
                continue
            split = parts[1]  # train, val, test
            if split not in stats:
                continue

            img_filename = parts[3]
            stem = Path(img_filename).stem
            lbl_entry = f"RDD_SPLIT/{split}/labels/{stem}.txt"

            # 1. Extract image
            dst_img_path = dst_root / "images" / split / img_filename
            with z.open(img_entry) as src, open(dst_img_path, "wb") as dst:
                dst.write(src.read())

            stats[split]["total_images"] += 1

            # 2. Extract and filter label
            pothole_lines = []
            try:
                raw_label_bytes = z.read(lbl_entry)
                lines = raw_label_bytes.decode("utf-8").strip().splitlines()
                for line in lines:
                    toks = line.strip().split()
                    if not toks:
                        continue
                    # Class 4 is Pothole (D40)
                    if toks[0] == "4":
                        # Map to Class 0: pothole
                        pothole_line = f"0 {' '.join(toks[1:])}"
                        pothole_lines.append(pothole_line)
            except KeyError:
                # No label file in zip -> purely negative sample
                pass

            dst_lbl_path = dst_root / "labels" / split / f"{stem}.txt"
            with open(dst_lbl_path, "w", encoding="utf-8") as lf:
                if pothole_lines:
                    lf.write("\n".join(pothole_lines) + "\n")
                    stats[split]["pothole_images"] += 1
                    stats[split]["pothole_boxes"] += len(pothole_lines)
                else:
                    # Write empty file to represent clean negative sample
                    lf.write("")
                    stats[split]["negative_images"] += 1

            if idx % 1000 == 0 or idx == len(india_images):
                print(f"Extracted {idx}/{len(india_images)} images ({stats['train']['total_images']} train, {stats['val']['total_images']} val, {stats['test']['total_images']} test)...")

    # Generate dataset.yaml
    dataset_yaml_path = dst_root.parent / "dataset.yaml"
    yaml_data = {
        "path": dst_root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "pothole"},
    }

    with open(dataset_yaml_path, "w", encoding="utf-8") as yf:
        yaml.dump(yaml_data, yf, default_flow_style=False)

    duration = time.time() - start_time
    print("\n" + "=" * 60)
    print(" [Success] India Dataset Extracted & Converted Successfully")
    print("=" * 60)
    print(f"Time Taken: {duration:.1f}s")
    for s, st in stats.items():
        print(f" Split: {s.upper()}")
        print(f"   Total Images     : {st['total_images']}")
        print(f"   Pothole Images   : {st['pothole_images']}")
        print(f"   Pothole Boxes    : {st['pothole_boxes']}")
        print(f"   Negative Images  : {st['negative_images']}")
    print(f"Config YAML written to: {dataset_yaml_path.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    extract_and_prepare_india_subset()
