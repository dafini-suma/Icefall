#!/usr/bin/env python


"""Parses DefinedCrowd datasets into Lhotse format.

Uses combined.tsv from the dataset directory as input and prepares recording
and supervision manifests in the specified output directory.
"""

import argparse
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSet, SupervisionSegment
import logging
from normalizer import Normalizer
from data_parser import DataParser
from pathlib import Path


class DefinedCrowdDataParser(DataParser):
    def __init__(self, input_file: str, output_dir: str, delimiter: str = '\t',
                 audio_parent_dir: str = '', language: str = 'English',
                 upper_case: bool = False):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.delimiter = delimiter
        if audio_parent_dir != '':
            self.audio_parent_dir = Path(audio_parent_dir)
        else:
            self.audio_parent_dir = self.input_file.parent
        self.language = language
        self.upper_case = upper_case
        self._fields = ['Channel', 'StartTime', 'EndTime', 'Transcription',
                        'RecordingId', 'RelativeFileName']

        self.normalizer = Normalizer(upper_case=self.upper_case)

        self.recording_ids = set()
        self.indices = {}
        self.parse()

    def _hms_to_s(self, hms: str) -> float:
        h, m, s = map(float, hms.split(':'))
        return 3600 * h + 60 * m + s

    def _s_to_str(self, s: float) -> str:
        return f'{1000 * s:08.0f}'

    def read_header_line(self, line: str):
        line = line.strip('\n').split(self.delimiter)
        _error = False
        for field in self._fields:
            try:
                self.indices[field] = line.index(field)
            except ValueError:
                logging.error(f'Unable to find {field} in the header')
                _error = True
        if _error:
            raise ValueError('Missing fields (see above).')
        self.total_fields = len(line)
        logging.info(f'Fields: {self.indices}, total: {self.total_fields}')

    def parse(self):
        self.output_dir.mkdir(exist_ok=True)
        utterances_parsed = set()
        with open(self.input_file) as input_file, RecordingSet.open_writer(
                self.output_dir / 'recordings.jsonl.gz') as \
                recording_file, SupervisionSet.open_writer(self.output_dir / \
                'supervisions.jsonl.gz') as supervision_file:
            self.read_header_line(input_file.readline())

            for line in input_file:
                line = line.strip('\n').split(self.delimiter)

                if len(line) != self.total_fields:
                    logging.warning('Skipping line with fewer fields:\n' +
                                    ' '.join(line))
                    continue

                channel = int(line[self.indices['Channel']]) - 1  # Zero-index
                start_time = self._hms_to_s(line[self.indices['StartTime']])
                end_time = self._hms_to_s(line[self.indices['EndTime']])
                duration = end_time - start_time
                transcription = line[self.indices['Transcription']]
                file_name = line[self.indices['RelativeFileName']]
                file_name = self.audio_parent_dir / file_name
                recording_id = line[self.indices['RecordingId']]
                speaker_id = f'{recording_id}-{channel}'
                utterance_id = (f'{speaker_id}_{self._s_to_str(start_time)}'
                                f'-{self._s_to_str(end_time)}')

                if utterance_id in utterances_parsed:
                    logging.warning(f'Duplicate utterance ID detected. '
                        f'Skipping {utterance_id}')
                    continue
                utterances_parsed.add(utterance_id)

                # Normalize transcription
                transcription = self.normalizer(transcription)

                if recording_id not in self.recording_ids:
                    self.recording_ids.add(recording_id)
                    recording = Recording.from_file(file_name)
                    recording_file.write(recording)

                segment = SupervisionSegment(
                            id=utterance_id,
                            recording_id=recording_id,
                            text=transcription,
                            channel=channel,
                            start=start_time,
                            duration=duration,
                            speaker=speaker_id,
                            language=self.language
                        )
                supervision_file.write(segment)

        logging.info('Parsing complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parses metadata.')
    parser.add_argument(
        '-i',
        '--input-file',
        help="""Input file to parse.
            It is generally named combined.tsv in DefinedCrowd datasets""",
        required=True
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        help='Output directory',
        required=True
    )
    parser.add_argument(
        '-a',
        '--audio-parent-dir',
        help='Audio parent directory. Defaults to dirname of the input file',
        default=''
    )
    parser.add_argument(
        '-d',
        '--delimiter',
        default='\t',
        help='Delimiter'
    )
    parser.add_argument(
        '-l',
        '--language',
        default='English',
        help='Language information for supervision metadata'
    )
    parser.add_argument(
        '-u',
        '--upper-case',
        action='store_true',
        help='Output supervisions in upper case'
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    DefinedCrowdDataParser(args.input_file, args.output_dir, args.delimiter,
                   audio_parent_dir=args.audio_parent_dir,
                   language=args.language, upper_case=args.upper_case)
