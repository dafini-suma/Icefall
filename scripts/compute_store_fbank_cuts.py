#!/usr/bin/env python

"""Prepares lhotse feature cuts"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from lhotse.cut import CutSet
from lhotse.features import Fbank, FbankConfig, ChunkedLilcomHdf5Writer
import logging
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path


import time
import torch


def compute_store_fbank_cuts(args: argparse.Namespace):
    if args.output_cuts.is_file():        
        logging.warning(
            f'Output cuts {args.output_cuts} already exist. Skipping feature extraction.')
        return

    def extract(cuts: CutSet, feature_dir: Path, queue: Queue) -> None:
        with ProcessPoolExecutor(args.num_threads) as ex:
            cuts = cuts.compute_and_store_features(
                Fbank(FbankConfig(sampling_rate=args.sampling_rate,
                                num_mel_bins=args.num_mel_bins)),
                storage_path=feature_dir,
                storage_type=ChunkedLilcomHdf5Writer,
                num_jobs=args.num_threads,
                executor=ex)
        queue.put(cuts)

    cuts = CutSet.from_file(args.input_cuts)
    logging.info(f'Found {len(cuts)} cuts')

    if args.num_threads > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    cuts = cuts.to_eager()
    args.feature_dir.mkdir(exist_ok=True)
    if args.num_jobs == 1:
        extract(cuts, args.feature_dir)
    else:
        if args.num_jobs > cpu_count():
            logging.warning(f'Limiting #jobs to CPU count {cpu_count()}')
            args.num_jobs = cpu_count()

        queue = Queue()
        processes = []
        cuts_split = cuts.split(args.num_jobs)
        for i, cut_subset in enumerate(cuts_split):
            p = Process(target=extract, args=(
                cut_subset, args.feature_dir / f"split-{i}", queue))
            p.start()
            processes.append(p)

        while queue.qsize() < args.num_jobs:
            time.sleep(5)
        
        cuts = reduce(lambda x, y: x + y, [
            queue.get() for _ in range(args.num_jobs)])

    args.output_cuts.parent.mkdir(exist_ok=True)
    cuts.to_file(args.output_cuts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepares lhotse features')
    parser.add_argument(
        '-i',
        '--input-cuts',
        type=Path,
        required=True,
        help='Input cut path'
    )
    parser.add_argument(
        '-o',
        '--output-cuts',
        type=Path,
        required=True,
        help='Output cut path'
    )
    parser.add_argument(
        '-f',
        '--feature-dir',
        type=Path,
        required=True,
        help='Output feature directory'
    )
    parser.add_argument(
        '-s',
        '--sampling-rate',
        type=int,
        default=8000,
        help='Sampling rate'
    )
    parser.add_argument(
        '-j',
        '--num-jobs',
        default=min(10, cpu_count()),
        type=int,
        help='Number of jobs to run'
    )
    parser.add_argument(
        '-t',
        '--num-threads',
        default=1,
        type=int,
        help='Number of threads for each job (recommended: 1)'
    )
    parser.add_argument(
        '-n',
        '--num-mel-bins',
        type=int,
        default=80,
        help='Number of Mel warped filters'
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    compute_store_fbank_cuts(args)
