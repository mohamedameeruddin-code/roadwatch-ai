"""
Official RDD2022 India Dataset Preparation Pipeline.
Converts Pascal VOC XML annotations to YOLO format, filters D40 (pothole),
preserves negative samples, executes reproducible 70:15:15 splitting without leakage,
generates dataset.yaml, and validates all paths.
"""

from collections import Counter
import json
import os
from pathlib import Path
import random
import shutil
import time
import xml.etree.ElementTree as ET
import yaml


def prepare_official_rdd2022_india(
    raw_dir: str = "data/raw/RDD2022_India",
    processed_dir: str = "data/processed",
    split_ratios=(0.70, 0.15, 0.15),
    seed: int = 42,
):
    start_time = time.time()
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)

    xml_dir = raw_path / "annotations" / "xmls"
    img_dir = raw_path / "images"

    if not xml_dir.is_dir() or not img_dir.is_dir():
        raise FileNotFoundError(f"Missing expected subdirectories in {raw_path}")

    xml_files = sorted(list(xml_dir.glob("*.xml")))
    img_files = sorted(list(img_dir.glob("*.jpg")))

    print("\n" + "=" * 65)
    print(" RDD2022 India Official Dataset Preparation Pipeline")
    print("=" * 65)
    print(f" Raw Directory       : {raw_path.resolve()}")
    print(f" Total XMLs Found    : {len(xml_files)}")
    print(f" Total Images Found  : {len(img_files)}")
    print(f" Target Class        : D40 -> 'pothole' (Class ID: 0)")
    print(f" Split Ratio         : {split_ratios} (Train / Val / Test)")
    print(f" Random Seed         : {seed} (Reproducible)")
    print("=" * 65 + "\n")

    # Step 1: Pair images and XMLs and parse D40 annotations
    paired_data = []
    pothole_items = []
    negative_items = []
    global_class_counts = Counter()
    total_d40_boxes = 0

    for xml_p in xml_files:
        stem = xml_p.stem
        img_p = img_dir / f"{stem}.jpg"
        if not img_p.is_file():
            print(f"[Warning] Corresponding image not found for {xml_p.name}")
            continue

        tree = ET.parse(xml_p)
        root = tree.getroot()

        size_elem = root.find("size")
        if size_elem is not None:
            w_elem = size_elem.find("width")
            h_elem = size_elem.find("height")
            img_w = int(w_elem.text) if w_elem is not None and w_elem.text else 0
            img_h = int(h_elem.text) if h_elem is not None and h_elem.text else 0
        else:
            img_w, img_h = 0, 0

        # Fallback to standard RDD image size if missing from XML header
        if img_w <= 0 or img_h <= 0:
            img_w, img_h = 720, 720

        d40_boxes = []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            global_class_counts[name] += 1
            if name == "D40":
                bndbox = obj.find("bndbox")
                if bndbox is not None:
                    xmin = int(float(bndbox.find("xmin").text))
                    ymin = int(float(bndbox.find("ymin").text))
                    xmax = int(float(bndbox.find("xmax").text))
                    ymax = int(float(bndbox.find("ymax").text))

                    # Sanity check
                    if xmax > xmin and ymax > ymin:
                        # YOLO format: 0 xc yc w h (normalized)
                        xmin_c = max(0, min(xmin, img_w))
                        xmax_c = max(0, min(xmax, img_w))
                        ymin_c = max(0, min(ymin, img_h))
                        ymax_c = max(0, min(ymax, img_h))

                        box_w = (xmax_c - xmin_c) / img_w
                        box_h = (ymax_c - ymin_c) / img_h
                        xc = (xmin_c + (xmax_c - xmin_c) / 2.0) / img_w
                        yc = (ymin_c + (ymax_c - ymin_c) / 2.0) / img_h

                        d40_boxes.append((xc, yc, box_w, box_h))
                        total_d40_boxes += 1

        item = {
            "stem": stem,
            "img_path": img_p,
            "boxes": d40_boxes,
            "has_pothole": len(d40_boxes) > 0,
        }

        if item["has_pothole"]:
            pothole_items.append(item)
        else:
            negative_items.append(item)

    print(f"Extracted {total_d40_boxes} D40 pothole bounding boxes across {len(pothole_items)} images.")
    print(f"Preserved {len(negative_items)} images without potholes as negative training samples.")
    print(f"All defect classes in original XMLs: {dict(global_class_counts)}\n")

    # Step 2: Stratified Split (70:15:15) to guarantee balanced defect distribution
    random.seed(seed)
    random.shuffle(pothole_items)
    random.shuffle(negative_items)

    n_p_train = int(len(pothole_items) * split_ratios[0])
    n_p_val = int(len(pothole_items) * split_ratios[1])

    n_n_train = int(len(negative_items) * split_ratios[0])
    n_n_val = int(len(negative_items) * split_ratios[1])

    splits = {
        "train": pothole_items[:n_p_train] + negative_items[:n_n_train],
        "val": pothole_items[n_p_train : n_p_train + n_p_val] + negative_items[n_n_train : n_n_train + n_n_val],
        "test": pothole_items[n_p_train + n_p_val :] + negative_items[n_n_train + n_n_val :],
    }

    # Verify zero data leakage
    train_stems = {x["stem"] for x in splits["train"]}
    val_stems = {x["stem"] for x in splits["val"]}
    test_stems = {x["stem"] for x in splits["test"]}

    assert len(train_stems.intersection(val_stems)) == 0, "Data leakage detected between train and val!"
    assert len(train_stems.intersection(test_stems)) == 0, "Data leakage detected between train and test!"
    assert len(val_stems.intersection(test_stems)) == 0, "Data leakage detected between val and test!"

    # Step 3: Write out to data/processed/
    split_stats = {}
    for split_name, items in splits.items():
        img_out = proc_path / "images" / split_name
        lbl_out = proc_path / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        num_pothole_imgs = 0
        num_boxes = 0

        for it in items:
            stem = it["stem"]
            dst_img = img_out / f"{stem}.jpg"
            dst_lbl = lbl_out / f"{stem}.txt"

            # Copy image if not already identical
            if not dst_img.is_file():
                shutil.copyfile(it["img_path"], dst_img)
            else:
                try:
                    if not os.path.samefile(it["img_path"], dst_img):
                        shutil.copyfile(it["img_path"], dst_img)
                except Exception:
                    pass

            # Write YOLO labels
            lines = []
            for xc, yc, bw, bh in it["boxes"]:
                lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                num_boxes += 1

            if lines:
                num_pothole_imgs += 1
                with open(dst_lbl, "w", encoding="utf-8") as lf:
                    lf.write("\n".join(lines) + "\n")
            else:
                # Negative sample: empty label file
                with open(dst_lbl, "w", encoding="utf-8") as lf:
                    lf.write("")

        split_stats[split_name] = {
            "total_images": len(items),
            "pothole_images": num_pothole_imgs,
            "pothole_boxes": num_boxes,
            "negative_images": len(items) - num_pothole_imgs,
        }

    # Step 4: Generate dataset.yaml
    yaml_path = proc_path.parent / "dataset.yaml"
    abs_processed = proc_path.resolve().as_posix()
    dataset_cfg = {
        "path": abs_processed,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "pothole"},
    }

    with open(yaml_path, "w", encoding="utf-8") as yf:
        yaml.dump(dataset_cfg, yf, default_flow_style=False)

    # Step 5: Save Preparation Summary
    summary = {
        "dataset_name": "RDD2022_India",
        "raw_source_directory": str(raw_path.resolve()),
        "processed_directory": str(proc_path.resolve()),
        "total_source_xmls": len(xml_files),
        "total_source_images": len(img_files),
        "target_class_code": "D40",
        "target_class_name": "pothole",
        "total_d40_boxes": total_d40_boxes,
        "total_pothole_images": len(pothole_items),
        "total_negative_images": len(negative_items),
        "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "random_seed": seed,
        "data_leakage_check": "PASSED (0 overlapping image stems)",
        "splits": split_stats,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }

    summary_file = proc_path.parent / "dataset_preparation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=4)

    print("\n" + "=" * 65)
    print(" Dataset Preparation Completed Successfully")
    print("=" * 65)
    for s_name, st in split_stats.items():
        print(f" {s_name.upper()} SPLIT:")
        print(f"   Total Images     : {st['total_images']}")
        print(f"   Pothole Images   : {st['pothole_images']}")
        print(f"   Pothole Boxes    : {st['pothole_boxes']}")
        print(f"   Negative Images  : {st['negative_images']}")
    print(f"\n Config YAML written to: {yaml_path.resolve()}")
    print(f" Summary JSON written to: {summary_file.resolve()}")
    print("=" * 65 + "\n")

    return summary


if __name__ == "__main__":
    prepare_official_rdd2022_india()
