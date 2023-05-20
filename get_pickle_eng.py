import pickle
from argparse import ArgumentParser
import json

import numpy as np
import pandas as pd

from pathlib import Path
import soundfile as sf

from tqdm import trange
import json
from pathlib import Path


def find_list(path_to_TIMIT):
    # creating relatives paths for test data set, validation data set and train data set
    test_phn_list = []
    train_phn_list = []
    train_dir = path_to_TIMIT
    train_dir = train_dir

    phn_list = [x for x in train_dir.iterdir() if x.name.endswith('phn')]
    for sentence in phn_list:
        train_phn_list.append(sentence)

    index: np.ndarray = np.arange(len(train_phn_list), dtype=int)
    sample_indices = np.random.choice(index,
                                      replace=False,
                                      size=int(len(train_phn_list) / 5))
    sample_indices = np.array(sample_indices, dtype=int)
    train_phn_list = np.array(train_phn_list)
    test_phn_list = train_phn_list[sample_indices]
    diff_indices = np.array(
        [x for x in set(index).difference(set(sample_indices))], dtype=int)

    return test_phn_list, train_phn_list[diff_indices]


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
    pickle_dict = {"VN": {}, "VT": {}}
    suffix = ".WAV"
    #suffix = suffix.lower()
    for k in trange(len(path_list), desc="test_wrd_list"):

        utterance = path_list[k]
        begin = []
        end = []
        phonemes = []

        wav_path = utterance.with_suffix(suffix)
        input_soundfile, sample_rate = sf.read(wav_path)

        with open(utterance) as phonme:
            data1 = phonme.readlines()

        for j in data1:
            begin, end, phoneme, label = j.strip().split(',')
            phonemes.append([
                int(begin.replace(".000000", "")),
                int(end.replace(".000000", "")), phoneme, label
            ])

        phonemes = pd.DataFrame(phonemes,
                                columns=['begin', 'end', 'phoneme', 'label'])
        phonemes['begin'] = phonemes['begin'].astype(int)
        phonemes['end'] = phonemes['end'].astype(int)
        phonemes['phoneme'] = phonemes['phoneme'].astype(str)
        phonemes['label'] = phonemes['label'].astype(str)

        print(phonemes)

        sound = phonemes
        """
        sound = phonemes[phonemes['begin'] >= begin]
        sound = sound[sound['end'] <= end]
        print(sound)
        input()
        """

        sec = list(sound["phoneme"])

        phone = "".join(sec[0:])
        begin_end_phonemes = sound.iloc[0:].to_numpy()

        #begin = begin_end_phonemes[0, 0]
        #end = begin_end_phonemes[-1, 1]
        begin_end_phonemes[:, :2] -= begin_end_phonemes[0, 0]
        if (begin_end_phonemes[0][-1] == "Vn"):
            if begin_end_phonemes[-1][-1] == "N":
                tag = "VN"
            else:
                tag = "VT"
        elif (begin_end_phonemes[0][-1] == "Vo"):
            if begin_end_phonemes[-1][-1] == "T":
                tag = "VT"
            else:
                tag = "VN"
        elif (begin_end_phonemes[0][-1] == "V"):
            if begin_end_phonemes[-1][-1] == "T" or "TC":
                tag = "VT"
            else:
                tag = "VN"
        else:
            raise Exception(
                f"Unexpected begin_end_phonemes[0][-1] {begin_end_phonemes[0][-1]}"
            )

        pickle_dict[tag][
            f"{utterance.parent}_{utterance.name}-{phone}-{end}{suffix}"] = {
                "Sound": input_soundfile,
                "Sample Rate": sample_rate,
                "Begin End Phonemes": begin_end_phonemes,
                "Phoneme Seq": phone,
            }

    print(
        f"VN Records: {len(pickle_dict['VN'])}, VT Records: {len(pickle_dict['VT'])}"
    )

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

    test_phn_list, train_phn_list = find_list(args.timit_directory, )

    find_data(
        test_phn_list,
        train_phn_list,
        args.output_dir,
    )
