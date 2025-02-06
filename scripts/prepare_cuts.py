#!/usr/bin/env python

"""Prepares lhotse feature cuts"""

import argparse
from lhotse.audio import RecordingSet
from lhotse.cut import CutSet, MonoCut
from lhotse.dataset.speech_recognition import validate_for_asr
from lhotse.kaldi import load_kaldi_data_dir
from lhotse.supervision import SupervisionSet
import logging
from pathlib import Path


def is_valid(cut: MonoCut) -> bool:
    try:
        validate_for_asr(CutSet.from_cuts([cut]))
        return True
    except AssertionError:
        return False


def prepare_cuts(args: argparse.Namespace):
    if (args.output_cut_dir / (args.cut_set_name + ".jsonl.gz")).is_file():
        logging.warning(
            'Output cuts already exist. Skipping feature extraction.')
        return
    if args.data_dir is not None:
        data_dir = args.data_dir.absolute()
        recordings = RecordingSet.from_file(
            data_dir / (args.recording_set_name + ".jsonl.gz"))
        supervisions = SupervisionSet.from_file(
            data_dir / (args.text_set_name + ".jsonl.gz"))
        if not all(r.sampling_rate == args.sampling_rate for r in recordings):
            recordings = recordings.resample(args.sampling_rate)
    else:
        data_dir = args.kaldi_data_dir.absolute()
        logging.warning(
            'It is recommended to use data_parser_kaldi.py for this purpose, '
            'since not all edge cases are handled in the current script.')
        recordings, supervisions, _ = load_kaldi_data_dir(
            data_dir, sampling_rate=args.sampling_rate)

    logging.info(f'Found {len(recordings)} recordings and {len(supervisions)}'
                 f' supervisions')

    cuts = CutSet.from_manifests(recordings=recordings,
                                 supervisions=supervisions)

    # The duration in the Kaldi "segments" file may have a slight mismatch with
    # the actual duration of the audio, typically by a very small value. We
    # automatically correct it here.
    for cut in cuts:
        dur_diff = cut.supervisions[-1].end - cut.duration
        if dur_diff > 1.0:
            logging.error(f'Segment information incorrect for cut {cut.id}')
        if dur_diff > 0.0:
            logging.warning(
                f'Adjusting segmentation for {cut.id} by {dur_diff}')
            cut.supervisions[-1].duration -= dur_diff

    # Some cuts fail the "trim_to_supervision" step. This could be due to
    # incorrect segments file or other reasons (that are not yet figured out).
    # We recursively look for the "failed" cuts and remove them.
    while True:
        cuts_trimmed = cuts.trim_to_supervisions(keep_overlapping=False)
        num_cuts = 0
        try:
            for num_cuts, cut in enumerate(cuts_trimmed, start=1):
                pass
            break
        except AssertionError:
            if num_cuts == 0:
                rec_idx = 0
            else:
                passed_id = cuts_trimmed[num_cuts - 1].recording.id
                for rec_idx, cut in enumerate(cuts, start=1):
                    if cut.recording.id == passed_id:
                        break
            error_cut = cuts[rec_idx]
            logging.warning(
                f'Failed to trim the recording ID {error_cut.recording.id} '
                f'to supervisions. Omitting it and proceeding.')
            cuts = cuts.filter(lambda x: x != error_cut)
    cuts = cuts_trimmed
    logging.info(f'Number of cuts: {num_cuts}')

    if args.validate_for_asr:
        cuts = cuts.filter(is_valid)
        num_cuts = sum(1 for cut in cuts)
        logging.info(f'Number of valid cuts: {num_cuts}')

    if args.perturb_speed:
        cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1)
        num_cuts = sum(1 for cut in cuts)
        logging.info(f'Number of cuts after speed perturbation: {num_cuts}')

    if args.min_cut_duration > 0:
        logging.info(f'Discarding cuts shorter than {args.min_cut_duration} '
                     f'second(s)')
        cuts = cuts.filter(lambda cut: all([
            sup.duration >= args.min_cut_duration
            for sup in cut.supervisions]))
        num_cuts = sum(1 for cut in cuts)
        logging.info(f'Final number of cuts: {num_cuts}')

    args.output_cut_dir.mkdir(exist_ok=True)
    cuts.to_jsonl(args.output_cut_dir / (args.cut_set_name + ".jsonl.gz"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepares lhotse features')
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument(
        '-d',
        '--data-dir',
        type=Path,
        default='.',
        help='Data directory containing {recordings,supervisions}.jsonl.gz'
    )
    group1.add_argument(
        '-k',
        '--kaldi-data-dir',
        type=Path,
        default='.',
        help='Kaldi data directory (not recommended. Use data_parser_kaldi.py)'
    )
    group1.required = True
    parser.add_argument(
        '-o',
        '--output-cut-dir',
        type=Path,
        required=True,
        help='Output directory to store cuts'
    )
    parser.add_argument(
        '-s',
        '--sampling-rate',
        type=int,
        default=8000,
        help='Sampling rate'
    )
    parser.add_argument(
        '-v',
        '--validate-for-asr',
        action='store_true',
        help='Validate cuts for ASR and retain only the valid ones'
    )
    parser.add_argument(
        '-m',
        '--min-cut-duration',
        type=float,
        default=1.0,
        help='Remove cuts shorter than the specified duration in seconds'
    )
    parser.add_argument(
        '-p',
        '--perturb-speed',
        action='store_true',
        help='Perturb speed. Recommended for training'
    )
    parser.add_argument(
        '-r',
        '--recording-set-name',
        type=str,
        default='recordings',
        help='Name of the recording set file'
    )
    parser.add_argument(
        '-t',
        '--text-set-name',
        type=str,
        default='supervisions',
        help='Name of the supervision set file'
    )
    parser.add_argument(
        '-c',
        '--cut-set-name',
        type=str,
        default='cuts',
        help='Name of the output cut set file'
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    prepare_cuts(args)
