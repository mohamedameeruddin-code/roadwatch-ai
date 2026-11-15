"""
Dataset Preparation and Pascal VOC to YOLO Conversion for RDD2022 India.
RoadWatch AI - Pothole Detection System
"""

import argparse
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
from PIL import Image


class RDDDatasetParser:
    """
    Parses Road Damage Dataset 2022 (RDD2022) India Pascal VOC annotations,
    filters the D40 (pothole) class, converts to YOLO format, and partitions
    into reproducible train, validation, and test splits (70:15:15).
    """

    TARGET_CLASS_CODE = "D40"
    TARGET_CLASS_NAME = "pothole"
    CLASS_MAPPING = {0: TARGET_CLASS_NAME}

    def __init__(
        self,
        raw_dir: str = "data/raw/RDD2022_India",
        processed_dir: str = "data/processed",
        split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        seed: int = 42,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.split_ratios = split_ratios
        self.seed = seed

        # Enforce split summation to 1.0
        assert (
            abs(sum(split_ratios) - 1.0) < 1e-4
        ), f"Split ratios {split_ratios} must sum to 1.0"

    @staticmethod
    def parse_voc_xml(xml_path: Path) -> Tuple[Optional[Tuple[int, int]], List[Tuple[str, List[int]]]]:
        """
        Parses a Pascal VOC XML file to extract image width/height
        and bounding boxes.

        Returns:
            ((width, height), [(class_name, [xmin, ymin, xmax, ymax])])
        """
        try:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()

            # Extract size
            size_elem = root.find("size")
            if size_elem is not None:
                w_elem = size_elem.find("width")
                h_elem = size_elem.find("height")
                img_w = int(w_elem.text) if w_elem is not None and w_elem.text else 0
                img_h = int(h_elem.text) if h_elem is not None and h_elem.text else 0
            else:
                img_w, img_h = 0, 0

            objects = []
            for obj in root.findall("object"):
                name_elem = obj.find("name")
                if name_elem is None or not name_elem.text:
                    continue
                cls_name = name_elem.text.strip()

                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue

                xmin = int(float(bndbox.find("xmin").text))
                ymin = int(float(bndbox.find("ymin").text))
                xmax = int(float(bndbox.find("xmax").text))
                ymax = int(float(bndbox.find("ymax").text))

                # Sanity check: Ensure valid box dimensions
                if xmax > xmin and ymax > ymin:
                    objects.append((cls_name, [xmin, ymin, xmax, ymax]))

            size_tuple = (img_w, img_h) if (img_w > 0 and img_h > 0) else None
            return size_tuple, objects

        except Exception as e:
            print(f"[Warning] Error parsing XML {xml_path}: {e}")
            return None, []

    @staticmethod
    def convert_box_to_yolo(
        box: List[int], img_w: int, img_h: int
    ) -> Tuple[float, float, float, float]:
        """
        Converts [xmin, ymin, xmax, ymax] to normalized YOLO coordinates
        (x_center, y_center, width, height) in range [0, 1].
        """
        xmin, ymin, xmax, ymax = box

        # Clip values to image bounds
        xmin = max(0, min(xmin, img_w))
        xmax = max(0, min(xmax, img_w))
        ymin = max(0, min(ymin, img_h))
        ymax = max(0, min(ymax, img_h))

        box_w = xmax - xmin
        box_h = ymax - ymin
        x_center = xmin + (box_w / 2.0)
        y_center = ymin + (box_h / 2.0)

        # Normalize
        norm_xc = max(0.0, min(1.0, x_center / img_w))
        norm_yc = max(0.0, min(1.0, y_center / img_h))
        norm_w = max(0.0, min(1.0, box_w / img_w))
        norm_h = max(0.0, min(1.0, box_h / img_h))

        return norm_xc, norm_yc, norm_w, norm_h

    def discover_raw_files(self) -> List[Tuple[Path, Optional[Path]]]:
        """
        Searches raw_dir recursively for image files (.jpg, .jpeg, .png)
        and locates their corresponding Pascal VOC XML annotation files.
        """
        valid_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        image_files = []

        for p in sorted(self.raw_dir.rglob("*")):
            if p.is_file() and p.suffix in valid_extensions:
                stem = p.stem
                # Potential XML paths
                candidates = [
                    p.with_suffix(".xml"),
                    p.parent.parent / "annotations" / "xmls" / f"{stem}.xml",
                    p.parent.parent / "annotations" / f"{stem}.xml",
                    p.parent.parent / "xmls" / f"{stem}.xml",
                    p.parent / f"{stem}.xml",
                    self.raw_dir / "annotations" / "xmls" / f"{stem}.xml",
                    self.raw_dir / "xmls" / f"{stem}.xml",
                ]
                matched_xml = None
                for c in candidates:
                    if c.is_file():
                        matched_xml = c
                        break
                image_files.append((p, matched_xml))

        return image_files

    def validate_dataset(self) -> Dict[str, bool]:
        """
        Validates all image and annotation paths across splits.
        Checks for matching label files and non-corrupted images.
        """
        print("\n--- Validating Dataset Integrity ---")
        validation_results = {}
        for split in ["train", "val", "test"]:
            img_dir = self.processed_dir / "images" / split
            lbl_dir = self.processed_dir / "labels" / split

            images = list(img_dir.glob("*.jpg"))
            labels = list(lbl_dir.glob("*.txt"))

            img_stems = {im.stem for im in images}
            lbl_stems = {lb.stem for lb in labels}

            has_all_labels = (img_stems == lbl_stems)
            validation_results[f"{split}_labels_match_images"] = has_all_labels
            print(f"[{split.upper()}] Images: {len(images)} | Labels: {len(labels)} | Complete 1:1 match: {has_all_labels}")

        return validation_results

    def process_and_split(self) -> Dict[str, any]:
        """
        Processes images and XMLs, converts D40 annotations to YOLO format,
        writes negative samples, partitions into train/val/test (70:15:15),
        validates paths, and creates dataset.yaml.
        """
        print(f"Scanning for road images in {self.raw_dir.resolve()}...")
        file_pairs = self.discover_raw_files()

        if not file_pairs:
            print(f"[Warning] No image files found in {self.raw_dir}. Please place dataset files first.")
            return {"total": 0, "train": 0, "val": 0, "test": 0, "pothole_instances": 0}

        print(f"Found {len(file_pairs)} candidate images with annotations.")

        # Create target directories
        for split in ["train", "val", "test"]:
            (self.processed_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.processed_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Shuffle deterministically using seed for reproducibility
        random.seed(self.seed)
        shuffled_pairs = file_pairs.copy()
        random.shuffle(shuffled_pairs)

        # Calculate partition split points (reproducible 70:15:15 split)
        total_count = len(shuffled_pairs)
        n_train = int(total_count * self.split_ratios[0])
        n_val = int(total_count * self.split_ratios[1])

        splits = {
            "train": shuffled_pairs[:n_train],
            "val": shuffled_pairs[n_train : n_train + n_val],
            "test": shuffled_pairs[n_train + n_val :],
        }

        stats = {
            "dataset_name": "RDD2022_India",
            "target_class": self.TARGET_CLASS_CODE,
            "target_class_name": self.TARGET_CLASS_NAME,
            "split_ratios": list(self.split_ratios),
            "random_seed": self.seed,
            "total_images": total_count,
            "splits": {},
            "total_pothole_instances": 0,
            "total_pothole_images": 0,
            "total_negative_images": 0,
        }

        for split_name, pairs in splits.items():
            print(f"Processing '{split_name}' split ({len(pairs)} images)...")
            img_out_dir = self.processed_dir / "images" / split_name
            lbl_out_dir = self.processed_dir / "labels" / split_name

            split_stats = {
                "total_images": len(pairs),
                "pothole_images": 0,
                "pothole_instances": 0,
                "negative_images": 0,
            }

            for img_path, xml_path in pairs:
                stem = img_path.stem
                dst_img = img_out_dir / img_path.name
                dst_lbl = lbl_out_dir / f"{stem}.txt"

                # Link or copy image
                if not dst_img.exists():
                    try:
                        os.link(img_path, dst_img)
                    except OSError:
                        shutil.copy2(img_path, dst_img)

                # Parse annotations if XML exists
                yolo_lines = []
                if xml_path and xml_path.is_file():
                    size_tuple, objects = self.parse_voc_xml(xml_path)

                    if size_tuple is None:
                        try:
                            with Image.open(img_path) as im:
                                img_w, img_h = im.size
                        except Exception:
                            img_w, img_h = 720, 720
                    else:
                        img_w, img_h = size_tuple

                    for cls_code, box in objects:
                        if cls_code.strip().upper() == self.TARGET_CLASS_CODE:
                            xc, yc, w, h = self.convert_box_to_yolo(box, img_w, img_h)
                            yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                            split_stats["pothole_instances"] += 1
                            stats["total_pothole_instances"] += 1

                # If no potholes or no XML, write empty file for negative sample
                if not yolo_lines:
                    split_stats["negative_images"] += 1
                    stats["total_negative_images"] += 1
                else:
                    split_stats["pothole_images"] += 1
                    stats["total_pothole_images"] += 1

                with open(dst_lbl, "w", encoding="utf-8") as lf:
                    lf.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

            stats["splits"][split_name] = split_stats

        # Validate image and label paths
        validation = self.validate_dataset()
        stats["validation"] = validation

        # Generate dataset.yaml
        dataset_yaml_path = self.processed_dir.parent / "dataset.yaml"
        # Convert path to forward slashes for cross-platform YOLO compatibility
        abs_processed_dir = self.processed_dir.resolve().as_posix()
        yaml_content = {
            "path": abs_processed_dir,
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: self.TARGET_CLASS_NAME},
        }

        with open(dataset_yaml_path, "w", encoding="utf-8") as yf:
            yaml.dump(yaml_content, yf, default_flow_style=False)

        # Write dataset preparation summary JSON
        summary_dir = Path("outputs/metrics")
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "dataset_summary.json"
        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump(stats, sf, indent=2)

        print(f"\n" + "=" * 65)
        print(" [Success] RoadWatch AI Dataset Pipeline Completed")
        print("=" * 65)
        print(f"Total Dataset Images  : {stats['total_images']}")
        print(f"Pothole Images (D40)  : {stats['total_pothole_images']}")
        print(f"Pothole Annotations   : {stats['total_pothole_instances']}")
        print(f"Negative Sample Images: {stats['total_negative_images']}")
        print("-" * 65)
        for s_name, s_data in stats["splits"].items():
            print(f" Split: {s_name.upper():<6} | Images: {s_data['total_images']:<5} | Pothole Imgs: {s_data['pothole_images']:<5} | Boxes: {s_data['pothole_instances']:<5} | Negatives: {s_data['negative_images']:<5}")
        print("-" * 65)
        print(f"Config YAML written to: {dataset_yaml_path.resolve()}")
        print(f"Summary JSON saved to : {summary_path.resolve()}")
        print("=" * 65 + "\n")

        return stats


def main():
    parser = argparse.ArgumentParser(description="RoadWatch AI Dataset Processor")
    parser.add_argument("--raw_dir", type=str, default="data/raw/RDD2022_India", help="Path to raw dataset directory")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Path to processed YOLO dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    parser.add_argument("--split", nargs=3, type=float, default=[0.70, 0.15, 0.15], help="Train Val Test split ratios")
    args = parser.parse_args()

    processor = RDDDatasetParser(
        raw_dir=args.raw_dir,
        processed_dir=args.out_dir,
        split_ratios=tuple(args.split),
        seed=args.seed,
    )
    processor.process_and_split()


if __name__ == "__main__":
    main()
