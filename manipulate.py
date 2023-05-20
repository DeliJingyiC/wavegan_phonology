import argparse
import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange
from wgan import get_manipulate_z, get_random_z

from wavegan import WaveGANDiscriminator, WaveGANGenerator, WaveGANQ


def save_samples(epoch_samples, epoch, output_dir, fs=16000):
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf
    """
    Save output samples to disk
    """
    sample_dir = output_dir
    sample_dir.mkdir(parents=True, exist_ok=True)

    for idx, samp in enumerate(epoch_samples):
        output_path = sample_dir / f"{epoch}_{idx + 1:02d}.wav"
        print(output_path)
        samp = samp[0]
        samp = (samp - np.mean(samp)) / np.abs(samp).max()
        plt.figure()
        plt.plot(samp)
        plt.savefig(Path(sample_dir) / f"{epoch}_{idx + 1:02d}.png")
        plt.close()
        sf.write(output_path, samp, fs)


def parse_arguments():
    def str_to_bool(flag: str):
        return {"true": True, "false": False}[flag.lower()]

    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Analyze a fiwGAN on a given latent code c')

    parser.add_argument('--model-size',
                        dest='model_size',
                        type=int,
                        default=64,
                        help='Model size parameter used in WaveGAN')
    parser.add_argument(
        '-ppfl',
        '--post-proc-filt-len',
        dest='post_proc_filt_len',
        type=int,
        default=512,
        help=
        'Length of post processing filter used by generator. Set to 0 to disable.'
    )
    parser.add_argument('--ngpus',
                        dest='ngpus',
                        type=int,
                        default=1,
                        help='Number of GPUs to use for training')
    parser.add_argument('--latent-dim',
                        dest='latent_dim',
                        type=int,
                        default=100,
                        help='Size of latent dimension used by generator')
    parser.add_argument('--verbose',
                        dest='verbose',
                        default=False,
                        action='store_true')
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to directory where model files will be output',
    )
    parser.add_argument(
        '--num_categ',
        dest='num_categ',
        type=int,
        default=3,
        help='Number of categorical variables',
    )
    parser.add_argument(
        '--model_path',
        dest='model_path',
        type=str,
        help="the path of the model",
    )
    parser.add_argument(
        '--random_range',
        dest='random_range',
        type=int,
        help="latent variable range",
    )
    parser.add_argument(
        '--num_epochs',
        dest='num_epochs',
        type=int,
        default=100,
        help='Number of epochs',
    )

    parser.add_argument('--job_id', type=str)

    parser.add_argument('--alter_axis', type=str)

    parser.add_argument('--alter_range', type=str)
    parser.add_argument('--filter_range', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--initialize_mode', type=str, default="uniform")
    parser.add_argument('--control_non_alter_vals',
                        type=str_to_bool,
                        default=True)
    parser.add_argument('--fix_laten_vals_accross_alter_vals',
                        type=str_to_bool,
                        default=True)

    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_arguments()

    with open(Path(args["output_dir"]) / f"params.json", 'w') as fout:
        json.dump(args, fout)
    latent_dim = args['latent_dim']
    ngpus = args['ngpus']
    model_size = args['model_size']
    Q_num_categ = args['num_categ']
    model_path = Path(args['model_path'])
    random_range = args['random_range']
    output_dir = Path(args['output_dir'])
    num_epochs = args['num_epochs']
    print(args['alter_axis'])
    alter_axis = [int(x) for x in args['alter_axis'].split(",")]
    filter_start, filter_end = [
        float(x) for x in args["filter_range"].split(",")
    ]
    alter_start, alter_end, interval = [
        float(x) for x in args["alter_range"].split(",")
    ]

    alter_vals = np.arange(alter_start, alter_end, interval)
    print(alter_vals)
    use_cuda = ngpus >= 1
    #load model
    model_gen = WaveGANGenerator(
        model_size=model_size,
        ngpus=ngpus,
        latent_dim=latent_dim,
        post_proc_filt_len=args['post_proc_filt_len'],
        upsample=True,
        verbose=args["verbose"],
    )
    model_gen.load_state_dict(torch.load(model_path / "Gen.pkl"))

    batch_size = args["batch_size"]
    # batch_size=1
    batch_step = 0
    (output_dir / "latent_v").mkdir(parents=True, exist_ok=True)

    for noise_v, altered_vals, i, total_batches in get_manipulate_z(
            Q_num_categ,
            batch_size,
            latent_dim,
            alter_axis=alter_axis,
            alter_vals=alter_vals,
            use_cuda=use_cuda,
            random_range=random_range,
            initialize_mode=args["initialize_mode"],
            control_non_alter_vals=args["control_non_alter_vals"],
            fix_laten_vals_accross_alter_vals=args[
                "fix_laten_vals_accross_alter_vals"],
    ):
        print(f"\rPrediction batch {i}/{total_batches}", end="")
        latent_v = noise_v.cpu().data.numpy()
        with open(
                output_dir / "latent_v" / f"{batch_step:02d}.pickle",
                'wb',
        ) as fout:
            pickle.dump(latent_v, fout)
        if use_cuda:
            noise_v = noise_v.cuda()
        # Generate outputs for fixed latent samples
        samp_output = model_gen.forward(noise_v)
        if use_cuda:
            samp_output = samp_output.cpu()

        samples = samp_output.data.numpy()
        save_samples(samples, f"{batch_step:02d}", output_dir / "Audio")

        batch_step += 1