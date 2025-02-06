#!/usr/bin/env python

from contextlib import nullcontext
import sys
import re


def parse(recog_file, ref_file, hyp_file, time_file=None):
    with open(recog_file) as f_in, open(ref_file, "w") as f_ref, \
            open(hyp_file, "w") as f_hyp, open(time_file, "w") if time_file \
            is not None else nullcontext() as f_tim:
        for line in f_in:
            line = re.sub(r"\[['\"]", " ", line)
            line = re.sub(r"['\"]\]", " ", line)
            line = re.sub("['\"], ['\"]", " ", line)
            if "\tref=" in line:
                line = re.sub(r":\s+ref=", " ", line)
                f_ref.write(line)
            elif "\thyp=" in line:
                line = re.sub(r":\s+hyp=", " ", line)
                f_hyp.write(line)
            elif f_tim is not None and "\ttimestamp_hyp=" in line:
                line = re.sub(r":\s+timestamp_hyp=", " ", line)
                f_tim.write(line)


if __name__ == "__main__":
    if len(sys.argv) not in [4, 5]:
        raise ValueError(
            'Expected 3 or 4 arguments. \n'
            'USAGE: parse_recog.py <recog-file> <ref-file> <hyp-file> '
            '[<timestamps_file>]')
    parse(*sys.argv[1:])
