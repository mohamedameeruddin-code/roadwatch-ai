"""
Unit tests for dataset parsing and YOLO conversion logic.
"""

from pathlib import Path
import tempfile
import unittest

from src.dataset import RDDDatasetParser


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Mock XML with one D40 (pothole) and one D00 (crack)
        self.mock_xml = """<annotation>
    <folder>images</folder>
    <filename>mock_001.jpg</filename>
    <size>
        <width>600</width>
        <height>400</height>
        <depth>3</depth>
    </size>
    <object>
        <name>D40</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>150</ymin>
            <xmax>300</xmax>
            <ymax>250</ymax>
        </bndbox>
    </object>
    <object>
        <name>D00</name>
        <bndbox>
            <xmin>400</xmin>
            <ymin>50</ymin>
            <xmax>500</xmax>
            <ymax>150</ymax>
        </bndbox>
    </object>
</annotation>"""
        self.xml_file = self.temp_path / "mock_001.xml"
        self.xml_file.write_text(self.mock_xml, encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_voc_xml(self):
        size_tuple, objects = RDDDatasetParser.parse_voc_xml(self.xml_file)
        self.assertEqual(size_tuple, (600, 400))
        self.assertEqual(len(objects), 2)
        # Verify first is D40
        self.assertEqual(objects[0][0], "D40")
        self.assertEqual(objects[0][1], [100, 150, 300, 250])

    def test_convert_box_to_yolo(self):
        box = [100, 150, 300, 250]  # w=200, h=100, xc=200, yc=200
        img_w, img_h = 600, 400

        xc, yc, w, h = RDDDatasetParser.convert_box_to_yolo(box, img_w, img_h)

        # Expected:
        # xc = 200 / 600 = 0.333333
        # yc = 200 / 400 = 0.5
        # w = 200 / 600 = 0.333333
        # h = 100 / 400 = 0.25
        self.assertAlmostEqual(xc, 200 / 600, places=4)
        self.assertAlmostEqual(yc, 200 / 400, places=4)
        self.assertAlmostEqual(w, 200 / 600, places=4)
        self.assertAlmostEqual(h, 100 / 400, places=4)


if __name__ == "__main__":
    unittest.main()
