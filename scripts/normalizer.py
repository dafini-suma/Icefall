#!/usr/bin/env python


""""Normalizes text"""


import argparse
from num2words import num2words
import re
import sys
from unidecode import unidecode  # !!! WARNING: This has GPL2 license !!!


class Normalizer:
    def __init__(self, map_file_in: str = None, upper_case=False):
        self._build_word_map(map_file_in)
        self.upper_case = upper_case
        self.currency_map = {
            '$': 'dollars',
            '€': 'euros',
            '£': 'pounds',
            '₹': 'rupees',
            '¥': 'yenn',
            '₩': 'won',
            '₱': 'peso',
            '₽': 'rubles',
            '฿': 'baht',
        }

    def _build_word_map(self, map_file_in: str) -> dict:
        self._DEFAULT_WORD_MAP = {
            '<sil>': '',
            '<babble>': '<unk>',
            '<uu>': '<unk>',
            '<vn>': '<unk>',
            '<filler>': '<unk>',
            '<noise>': '<unk>',
            '<non-speech>': '<unk>',
            '<inaudible>': '<unk>',
            '[laughter/]': '<unk>',
            '[emp/]': '<unk>',
            '[n_s/]': '<unk>',
            '[filler/]': '<unk>',
            '[fo/]': '<unk>',
            '[c_o/]': '<unk>',
            '[unintelligible/]': '<unk>',
            '[undefined/]': '<unk>',
            '[laughter]': '<unk>',
            '[noise]': '<unk>',
            '[vocalized-noise]': '<unk>',
            '[foreign language]': '<unk>',
            '[operator instructions]': '<unk>',
            '<p>': '',
            '<p/>': '',
            '<qj0>': '',
        }
        self.word_map = self._DEFAULT_WORD_MAP
        if map_file_in is not None:
            with open(map_file_in) as f:
                for line in f:
                    line = line.strip('\n').split(' ')
                    self.word_map[line[0]] = line[1]
        return self.word_map

    def has_digits(self, word: str) -> bool:
        return any(char.isdigit() for char in word)

    def normalize_word_with_digits(self, word: str) -> str:
        parts = re.split(
            r'([0-9]*1st)|([0-9]*2nd)|([0-9]*3rd)|([0-9]*[04-9]th)|([0-9]+)',
            word)
        parts = list(filter(None, parts))

        for index, part in enumerate(parts):
            if part.isdigit():
                # Special case for years
                if len(part) == 4 and (
                        (part.startswith('20') and not part.startswith('200'))
                        or part.startswith('19') or part.startswith('18')):
                    part = num2words(part[0:2]) + ' ' + num2words(part[2:4])
                else:
                    part = num2words(part)
                part = part.replace(',', '').replace('-', ' ')
                parts[index] = part
            elif self.has_digits(part):  # e.g. 1st, 235th.
                part = num2words(part[:-2], ordinal=True)
                part = part.replace(',', '').replace('-', ' ')
                parts[index] = part
        word = ' '.join(parts)

        return word

    def normalize(self, line: str) -> str:
        line = line.lower()

        # Special tags in the text are within <> or [].
        # Add space before <, [ and after >, ]
        line = re.sub(r'([^ ])<', r'\g<1> <', line)
        line = re.sub(r'>([^ ])', r'> \g<1>', line)
        line = re.sub(r'([^ ])\[', r'\g<1> [', line)
        line = re.sub(r'\]([^ ])', r'] \g<1>', line)

        # Remove comma if it appears between digits. E.g. 43,570 -> 43570
        # NOTE: This must be different for French
        line = re.sub(r'([0-9]),([0-9])', r'\g<1>\g<2>', line)

        # Handle currency symbols
        if re.search(r'[\$€£₹¥₩₱₽฿]\s*[0-9]+\.?[0-9]*', line) is not None:
            line = re.sub(
                r'([\$€£₹¥₩₱₽฿])\s*([0-9]+\.?[0-9]*) '
                r'(thousand|million|billion|trillion)',
                r'\g<2> \g<3> \g<1>',
                line)

            # Special case for the lines having no thousand/million/... suffix
            line = re.sub(
                r'([\$€£₹¥₩₱₽฿])\s*([0-9]+\.?[0-9]*)', r'\g<2> \g<1>', line)

            # Convert the symbol into correct word. Note: Unidecode, that is
            # used later, does expand currency symbols but uses abbreviated
            # forms. (e.g. € -> EUR)
            for sym in self.currency_map:
                line = line.replace(sym, self.currency_map[sym])

        # Decode non-ASCII characters
        line = unidecode(line)

        # Convert % character into percent, either appearing succeeding a
        # number, or standalone. E.g.
        # 1,234.5% -> 1,234.5 percent
        # 6 % -> 6 percent
        line = re.sub(r'([0-9\.,]*)%', r'\g<1> percent', line)

        # Normalize decimal point. E.g.: 1,234.56 -> 1,234 point 5 6
        # NOTE: This must be different for French
        line = re.sub(
            r'([0-9,]+)\.([0-9])([0-9]?)([0-9]?)([0-9]?)',
            r'\g<1> point \g<2> \g<3> \g<4> \g<5>',
            line)

        # Handle negative numbers
        line = re.sub(r'(^| )-([0-9\.]+)($| )', r'\g<1>minus \g<2>\g<3>', line)

        # Map words according to the word map
        word_list = line.split()
        for index, word in enumerate(word_list):
            while word in self.word_map:
                word = self.word_map[word]
            if word_list[index] != word:
                word_list[index] = word

        line = ' '.join(word_list)

        # Handle initials (e.g. C.C. Wei -> c c wei)
        line = re.sub(r'(^| )([A-Za-z])\.', r'\g<1>\g<2> ', line)

        # Handle web links (e.g. www.uniphore.com -> www dot uniphore dot com)
        line = re.sub(
            r'([A-Za-z0-9\-]*)\.([A-Za-z0-9\-])', r'\g<1> dot \g<2>', line)

        # Replace special characters with a space
        line = re.sub(r'[^A-Za-z0-9 \'<>\[\]\$]', ' ', line)

        # Remove single quote if between spaces
        line = re.sub(' \' ', ' ', line)

        # Normalize words with digits
        word_list = line.split()
        for index, word in enumerate(word_list):
            if self.has_digits(word):
                word = self.normalize_word_with_digits(word)
            if word_list[index] != word:
                word_list[index] = word

        line = ' '.join(word_list)
        if self.upper_case:
            line = line.upper()
        return line

    def __call__(self, line: str) -> str:
        return self.normalize(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Normalizes text.')
    parser.add_argument(
        '-m',
        '--map-file',
        default=None,
        help='Word map file'
    )
    parser.add_argument(
        '-u',
        '--upper-case',
        action='store_true',
        help='Output in upper case'
    )
    args = parser.parse_args()

    normalizer = Normalizer(args.map_file, args.upper_case)
    # inputs = [
    # r"1234 \$56,768 €2003 £36 ₹500", "Sales grew by 13% to €39.6 billion"]
    inputs = sys.stdin
    for line in inputs:
        line = line.strip('\n')
        sys.stdout.write(f'Original: {line}\n')
        sys.stdout.write(f'Normalized: {normalizer(line)}\n')
