#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys

import numpy as np

from PIL import Image
import labelme


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument(
        "--val_split", help="Plance every n-th element in the validaiton set", default=10, type=int)

    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)

    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "train"))
    os.makedirs(osp.join(args.output_dir, "val"))
    os.makedirs(osp.join(args.output_dir, "train", "images"))
    os.makedirs(osp.join(args.output_dir, "train", "labels"))
    os.makedirs(osp.join(args.output_dir, "val", "images"))
    os.makedirs(osp.join(args.output_dir, "val", "labels"))

    print("Creating dataset:", args.output_dir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)

    class_names = tuple(class_names)
    print("class_names:", class_names)

    image_no = 0
    for filename in glob.glob(osp.join(args.input_dir, "*.json")):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]

        if image_no % args.val_split == 0:
            out_img_file = osp.join(
                args.output_dir, "val", "images", base + ".jpg")
            out_lbl_file = osp.join(
                args.output_dir, "val", "labels", base + ".png")
        else:
            out_img_file = osp.join(
                args.output_dir, "train", "images", base + ".jpg")
            out_lbl_file = osp.join(
                args.output_dir, "train", "labels", base + ".png")

        image_no += 1

        with open(out_img_file, "wb") as f:
            f.write(label_file.imageData)

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        lbl, _ = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )

        im = Image.fromarray(lbl).convert('RGB')
        im.save(out_lbl_file)


if __name__ == "__main__":
    main()
