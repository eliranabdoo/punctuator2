import os
import re
import argparse
from typing import List

from sklearn.model_selection import train_test_split

from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_TEST_SIZE = 0.1

NUM = '<NUM>'
EOS_PUNCTS = {".": ".PERIOD", "?": "?QUESTIONMARK", "!": "!EXCLAMATIONMARK"}
INS_PUNCTS = {",": ",COMMA", ";": ";SEMICOLON", ":": ":COLON", "-": "-DASH"}

forbidden_symbols = re.compile(r"[\[\]\(\)\/\\\>\<\=\+\_\*]")
numbers = re.compile(r"\d")
multiple_punct = re.compile(r'([\.\?\!\,\:\;\-])(?:[\.\?\!\,\:\;\-]){1,}')

is_number = lambda x: len(numbers.sub("", x)) / len(x) < 0.6


def untokenize(line):
    return line.replace(" '", "'").replace(" n't", "n't").replace("can not", "cannot")


def skip(line):
    if line.strip() == '':
        return True

    last_symbol = line[-1]
    if not last_symbol in EOS_PUNCTS:
        return True

    if forbidden_symbols.search(line) is not None:
        return True

    return False


def process_line(line):
    tokens = word_tokenize(line)
    output_tokens = []

    for token in tokens:

        if token in INS_PUNCTS:
            output_tokens.append(INS_PUNCTS[token])
        elif token in EOS_PUNCTS:
            output_tokens.append(EOS_PUNCTS[token])
        elif is_number(token):
            output_tokens.append(NUM)
        else:
            output_tokens.append(token.lower())

    return untokenize(" ".join(output_tokens) + " ")


def annotate_puncts(data: str) -> List[str]:
    res = []
    for line in data.splitlines():
        line = line.replace("\"", "").strip()
        line = multiple_punct.sub(r"\g<1>", line)

        if not skip(line):
            line = process_line(line)
        res.append(line)
    return res


def float_range(mini, maxi):
    """Return function handle of an argument type function for
       ArgumentParser checking a float range: mini <= arg <= maxi
         mini - minimum acceptable argument
         maxi - maximum acceptable argument"""

    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""

        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < mini or f > maxi:
            raise argparse.ArgumentTypeError("must be in range [" + str(mini) + " .. " + str(maxi) + "]")
        return f

    # Return function handle to checking function
    return float_range_checker


def fix_misplaced_newlines(data: str) -> List[str]:
    return re.sub("[^.]\n", " ", data).splitlines()


def prepare_data(data_path, results_dir, validation_size, test_size, normalization_funcs):
    with open(data_path, 'r', encoding="utf-8") as f:
        data = f.read().splitlines()

    for normalization_func in normalization_funcs:
        data = normalization_func('\n'.join(data))

    rest, test = train_test_split(data, test_size=test_size)
    del data
    train, val = train_test_split(rest, test_size=validation_size)
    del rest

    data_filename = os.path.splitext(os.path.basename(data_path))[0]

    for suffix, dataset in {".train.txt": train,
                            ".dev.txt": val,
                            ".test.txt": test}.items():
        curr_path = os.path.join(results_dir, data_filename + suffix)
        print(f"Writing to {len(dataset)} lines to {curr_path}")
        with open(curr_path, "w", encoding="utf-8") as f:
            f.write('\n'.join(dataset))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--results-dir")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("-v", "--validation-size", type=float_range(0, 1), default=DEFAULT_VALIDATION_SIZE)
    parser.add_argument("-t", "--test-size", type=float_range(0, 1), default=DEFAULT_TEST_SIZE)
    args = parser.parse_args()

    paths = args.paths
    for path in paths:
        results_dir = args.results_dir
        if results_dir is None:
            results_dir = os.path.dirname(path)
        prepare_data(data_path=path, results_dir=results_dir,
                     validation_size=args.validation_size, test_size=args.test_size,
                     normalization_funcs=[fix_misplaced_newlines, annotate_puncts])


if __name__ == "__main__":
    main()
