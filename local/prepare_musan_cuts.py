#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This file prepares cuts of the musan dataset.
It looks for manifests in the directory data/manifests.
"""

import argparse
import logging
from pathlib import Path

from lhotse import CutSet, MonoCut, combine
from lhotse.recipes.utils import read_manifests_if_cached


def is_cut_long(c: MonoCut) -> bool:
    return c.duration > 5


def prepare_musan_cuts(args):
    dataset_parts = (
        "music",
        "speech",
        "noise",
    )
    prefix = "musan"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=args.manifest_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    musan_cuts_path = args.output_cut_dir / "musan_cuts.jsonl.gz"

    if musan_cuts_path.is_file():
        logging.info(f"{musan_cuts_path} already exists - skipping")
        return

    logging.info("Preparing Musan cuts")

    # create chunks of Musan with duration 5 - 10 seconds
    musan_cuts = (
        CutSet.from_manifests(
            recordings=combine(
                part["recordings"] for part in manifests.values())
        )
        .resample(args.sampling_rate)
        .cut_into_windows(10.0)
        .filter(is_cut_long)
    )
    musan_cuts.to_file(musan_cuts_path)
    logging.info(f'Successfully stored {musan_cuts_path}')


if __name__ == "__main__":
    format = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    parser = argparse.ArgumentParser(description='Prepares musan features')

    parser.add_argument(
        '-m',
        '--manifest-dir',
        type=Path,
        required=True,
        help='Manifest directory'
    )
    parser.add_argument(
        '-o',
        '--output-cut-dir',
        type=Path,
        required=True,
        help='Output cut directory'
    )
    parser.add_argument(
        '-s',
        '--sampling_rate',
        type=int,
        default=None,
        help='Musan sampling_rate'
    )
    args = parser.parse_args()
    logging.basicConfig(format=format, level=logging.INFO)
    prepare_musan_cuts(args)
