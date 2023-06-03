# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
"""Generate labeled and unlabeled dataset for coco train.

Example:
python tools/coco_semi.py
"""

import argparse
import numpy as np
import json
import os


def prepare_coco_data(input1, input2, output):
    """Prepare COCO dataset for Semi-supervised learning
    Args:
      seed: random seed for dataset split
      percent: percentage of labeled dataset
      version: COCO dataset version
    """

    anno1 = json.load(open(input1))
    anno2 = json.load(open(input2))

    len_image_list = len(anno1["images"]) + 1
    for img in anno2["images"]:
        img["id"] += len_image_list
        anno1["images"].append(img)

    len_label_list = len(anno1["annotations"]) + 1
    for label in anno2["annotations"]:
        label["id"] += len_label_list
        label["image_id"] += len_image_list
        anno1["annotations"].append(label)

    with open(output, "w") as f:
        json.dump(anno1, f)

    print(output)
    print(len_image_list, len_label_list)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input1", type=str)
    parser.add_argument("--input2", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    print(args)
    prepare_coco_data(args.input1, args.input2, args.output)
    print('ok.')
