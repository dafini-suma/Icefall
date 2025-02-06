#!/usr/bin/env python

"""Parses Kaldi directory and generates manifests.
This is a wrapper around lhotse's internal Kaldi loader."""


import argparse
from lhotse.kaldi import load_kaldi_data_dir
from lhotse.cut import CutSet
import logging
from data_parser import DataParser
from pathlib import Path
import shutil
from subprocess import Popen, PIPE


class KaldiDataParser(DataParser):
    def __init__(
            self, input_dir: str, output_dir: str, sampling_rate: int,
            kaldi_root_dir: str = '', store_cuts: bool = False,
            empty_token: bytes = b'!SIL', job_count: int = 1) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sampling_rate = sampling_rate
        if kaldi_root_dir:
            self.kaldi_root = Path(kaldi_root_dir)
        else:
            self.kaldi_root = None
        self.store_cuts = store_cuts
        self.empty_token = empty_token
        self.job_count = job_count

        assert self.input_dir.is_dir(), 'Input directory does not exist'
        self.frame_shift = 0.01  # Since we have not seen otherwise
        self._sox_options = [
            '-r', str(self.sampling_rate), '-t', 'wav', '-b', '16',
            '-e', 'signed-integer', '-', '|']

    def introduce_empty_tokens(self):
        (self.input_dir / 'text').rename(self.input_dir / 'text.bak')
        with open(self.input_dir / 'text.bak', 'rb') as f_in, open(
                self.input_dir / 'text', 'wb') as f_out:
            for line in f_in:
                line = line.split()
                if len(line) == 1:
                    line.append(self.empty_token)
                f_out.write(b' '.join(line) + b'\n')

    def _last_cmd_is_sox(self, line: list) -> bool:
        loc = self._find_max_loc(line, '|')
        return line[loc + 1] == 'sox'

    def _find_max_loc(self, line: list, token: str) -> int:
        try:
            return 1 + max(
                loc for loc, val in enumerate(line[1: -1]) if val == token)
        except ValueError:
            return 0

    def resample_wav_files(self):
        (self.input_dir / 'wav.scp').rename(self.input_dir / 'wav.scp.bak')
        with open(self.input_dir / 'wav.scp.bak') as f_in, open(
                self.input_dir / 'wav.scp', 'w') as f_out:
            for line in f_in:
                line = line.replace('|', ' | ')
                line = line.split()
                if len(line) == 2:
                    line = [line[0], 'sox', line[1]] + self._sox_options
                elif self._last_cmd_is_sox(line):
                    loc = self._find_max_loc(line, '|')
                    rel_resample_loc = line[
                        loc: -1].index('-r') if '-r' in line[loc: -1] else 0
                    if rel_resample_loc:
                        line[loc + rel_resample_loc + 1] = str(
                            self.sampling_rate)
                    else:
                        rel_hyphen_loc = line[
                            loc: -1].index('-') if '-' in line[loc: -1] else 0
                        line[loc + rel_hyphen_loc: loc + rel_hyphen_loc] = [
                            '-r', str(self.sampling_rate)]
                else:
                    line.extend(['sox', '-'] + self._sox_options)
                f_out.write(' '.join(line) + '\n')

    def parse(self):
        self.introduce_empty_tokens()
        # self.resample_wav_files()

        kaldi_utils_dir = Path.cwd() / 'utils'
        if not kaldi_utils_dir.is_dir():
            if self.kaldi_root is None:
                logging.error(
                    'Kaldi root is required to create utt2dur/segments files')
            kaldi_utils_dir.symlink_to(
                self.kaldi_root / 'egs' / 'wsj' / 's5' / 'utils')
        assert kaldi_utils_dir.is_dir(), 'Incorrect Kaldi utils directory'

        if not (self.input_dir / 'utt2dur').is_file():
            logging.info('Creating utt2dur file')
            Popen([
                kaldi_utils_dir / 'data' / 'get_utt2dur.sh',
                self.input_dir]).communicate()

        if not (self.input_dir / 'segments').is_file():
            logging.info('Creating segments file')

            logging.info('Creating segments file')
            with open(self.input_dir / 'segments', 'w') as f:
                Popen([
                    kaldi_utils_dir / 'data' / 'get_segments_for_data.sh',
                    self.input_dir],
                    stdout=f, stderr=PIPE).communicate()

        logging.info('Loading kaldi directory')
        #logging.info('- - - - - - - - - - - - - - - - -')
        #logging.info(f"{self.input_dir=} {self.sampling_rate=}")

        rec, sup, fea = load_kaldi_data_dir(
            self.input_dir,
            sampling_rate=self.sampling_rate,
            frame_shift=self.frame_shift,
            num_jobs=self.job_count)

        assert sup is not None, 'Supervisions not loaded'

        self.output_dir.mkdir(exist_ok=True)
        rec.to_jsonl(self.output_dir / 'recordings.jsonl.gz')
        sup.to_jsonl(self.output_dir / 'supervisions.jsonl.gz')

        if self.store_cuts:
            if fea is None:
                logging.error('Failed to load Kaldi features')

            # Make sure the feature and supervision timestamps match
            for f in fea:
                f.start = sup[f.storage_key].start
                sup[f.storage_key].duration = f.duration

            cuts = CutSet.from_manifests(
                recordings=rec,
                supervisions=sup,
                features=fea
            )
            cuts.to_jsonl(self.output_dir / 'cuts.jsonl.gz')

        # shutil.move(self.input_dir / 'wav.scp.bak', self.input_dir / 'wav.scp')
        # shutil.move(self.input_dir / 'text.bak', self.input_dir / 'text')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parses metadata.')
    parser.add_argument(
        '-i',
        '--input-dir',
        type=str,
        help='Input Kaldi directory to parse',
        required=True
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        help='Output directory',
        type=str,
        required=True
    )
    parser.add_argument(
        '-s',
        '--sampling-rate',
        default=16000,
        help='Sampling rate',
        type=int
    )
    parser.add_argument(
        '-k',
        '--kaldi-root-dir',
        default='',
        help='Kaldi root directory',
        type=str
    )
    parser.add_argument(
        '-c',
        '--store-cuts',
        help='Store cuts additionally (not recommended for model training)',
        action='store_true'
    )
    parser.add_argument(
        '-e',
        '--empty-token',
        default=b'!SIL',
        help='Token to replace empty transcriptions',
        type=bytes
    )
    parser.add_argument(
        '-j',
        '--job-count',
        default=1,
        type=int,
        help='Job count'
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    parser = KaldiDataParser(
        args.input_dir, args.output_dir, args.sampling_rate,
        args.kaldi_root_dir, args.store_cuts, args.empty_token,
        args.job_count)
    parser.parse()
