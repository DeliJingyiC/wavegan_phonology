import pickle
from argparse import ArgumentParser
import json

import numpy as np
import pandas as pd

from pathlib import Path
import soundfile as sf

from tqdm import trange
import json


def find_list(path_to_TIMIT:Path):
    # creating relatives paths for test data set, validation data set and train data set
    train_phn_list = []
    train_dir = path_to_TIMIT
    for part in train_dir.iterdir():
        if (part.is_file()):
            continue
        for subpart in part.iterdir():
            # print(subpart.name)
            if (subpart.is_file()):
                continue
            phn_list = subpart.iterdir()
            # print(phn_list)
            # input()
            phn_list = [x for x in phn_list if x.name.endswith('PHN') and "SA" not in x.name]
            train_phn_list.extend(iter(phn_list))
    # input(len(train_phn_list))
    return  train_phn_list


def find_data(
    test_phn_list,
    train_phn_list,
    output_dir,
):

    pickle_dict = convert_to_pkl(train_phn_list)

    with open(output_dir / "train.pkl", 'wb') as fout:
        pickle.dump(pickle_dict, fout)

    pickle_dict = convert_to_pkl(test_phn_list, )

    with open(output_dir / "test.pkl", 'wb') as fout:
        pickle.dump(pickle_dict, fout)


def convert_to_pkl(path_list):
    pickle_dict = {}
    suffix = ".WAV"
    # suffix = suffix.lower()   
    for k in trange(len(path_list), desc="test_wrd_list"):

        utterance = path_list[k]
        wav_path = utterance.with_suffix(suffix)
        input_soundfile, sample_rate = sf.read(wav_path)

        pickle_dict[f"{utterance.parent}_{utterance.name}"] = {
            "Sound": input_soundfile,
            "Sample Rate": sample_rate,
        }

    return pickle_dict


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--timit_directory',
                        type=str,
                        required=True,
                        help='Directory containing TIMIT dataset.',
                        metavar='<TimitDirectory>')

    parser.add_argument('--output_dir',
                        type=str,
                        required=True,
                        help='Target directory for the experiment',
                        metavar='<TargetDirectory>')
    parser.add_argument('--test_only',
                        default=False,
                        action='store_true',
                        help='Test already trained model')

    parser.add_argument('--job_id', type=str)
    parser.add_argument('--time_directory', type=str, required=True)

    args = parser.parse_args()

    with open(Path(args.output_dir) / "parameters.json", 'w') as fout:
        json.dump(args.__dict__, fout)

    args.timit_directory = Path(args.timit_directory)
    args.time_directory = Path(args.time_directory)
    args.output_dir = Path(args.output_dir)

    train_phn_list = find_list(args.timit_directory/"TRAIN", )
    test_phn_list = find_list(args.timit_directory/"TEST", )

    find_data(test_phn_list, train_phn_list, args.output_dir)
