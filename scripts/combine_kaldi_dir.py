#!/usr/bin/env python

"""Makes a single file out of Kaldi directory.
We will use the output file for publishing and to make lhotse manifests."""


import argparse
import logging
from pathlib import Path


def main(input_dir: str, output_file: str, delimiter: str) -> None:
    input_dir = Path(input_dir)

    # Populate recording ID to its wav file
    reco2files = {}
    pipe_warning =False
    with open(input_dir / 'wav.scp') as f:
        for line in f:
            line = line.split()
            reco_id = line[0]
            wav_file = ' '.join(line[1:])
            if wav_file.endswith('|'):
                wav_file = wav_file[:-1]
                pipe_warning = True
            reco2files[reco_id] = wav_file
    if pipe_warning:
        logging.warning(
            'wav.scp contains piped input. Better use data_parser_kaldi.py')

    # Populate utterance ID to its text
    utt2text = {}
    with open(input_dir / 'text') as f:
        for line in f:
            line = line.split()
            utt_id = line[0]
            text = ' '.join(line[1:])
            utt2text[utt_id] = text

    utt2spk = {}
    with open(input_dir / 'utt2spk') as f:
        for line in f:
            line = line.split()
            utt2spk[line[0]] = line[1]

    FIELDS = (
        'RecordingId', 'SpeakerId', 'UtteranceId', 'RelativeFileName',
        'Transcription', 'StartTime', 'EndTime'
        )
    with open(input_dir / 'segments') as f_in, open(output_file, 'w') as f_out:
        f_out.write(delimiter.join(FIELDS) + '\n')
        for line in f_in:
            utt_id, reco_id, start_time, end_time = line.split()
            f_out.write(delimiter.join([
                reco_id,
                utt2spk[utt_id],
                utt_id,
                reco2files[reco_id],
                utt2text[utt_id],
                start_time,
                end_time
                ]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parses Kaldi directory into a single delimited text.')
    parser.add_argument(
        '-i',
        '--input-dir',
        required=True,
        help='Input Kaldi style data directory'
    )
    parser.add_argument(
        '-o',
        '--output-file',
        required=True,
        help='Output file'
    )
    parser.add_argument(
        '-d',
        '--delimiter',
        default='\t',
        help='Delimiter. tab recommended, space strongly discouraged'
    )
    args = parser.parse_args()

    main(args.input_dir, args.output_file, args.delimiter)
