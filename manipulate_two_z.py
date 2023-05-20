from wgan import get_random_z, get_manipulate_z
import argparse
from wavegan import WaveGANDiscriminator, WaveGANGenerator, WaveGANQ
import os
from pathlib import Path
import torch
from tqdm import trange
import pickle
import numpy as np
import matplotlib.pyplot as plt


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
        output_path = sample_dir / f"{epoch}_{idx + 1}.wav"
        print(output_path)
        samp = samp[0]
        samp = (samp - np.mean(samp)) / np.abs(samp).max()
        plt.figure()
        plt.plot(samp)
        plt.savefig(Path(sample_dir) / f"{epoch}_{idx + 1:02d}.png")
        plt.close()
        sf.write(output_path, samp, fs)


def parse_arguments():
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

    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_arguments()

    latent_dim = args['latent_dim']
    ngpus = args['ngpus']
    model_size = args['model_size']
    model_dir = os.path.join(args['output_dir'], args["job_id"])
    args['model_dir'] = Path(model_dir)
    model_dir = args['model_dir']
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
    batch_size = len(alter_vals)
    #Starting: analyze the model
    for axis in alter_axis:
        for epoch in trange(num_epochs):
            noise_v, alter_vals = get_manipulate_z(
                Q_num_categ,
                batch_size,
                latent_dim,
                alter_axis=axis,
                alter_vals=alter_vals,
                use_cuda=use_cuda,
                random_range=random_range,
            )
            latent_v = noise_v.cpu().data.numpy()
            print(latent_v)
            (model_dir / "latent_v").mkdir(parents=True, exist_ok=True)
            with open(
                    model_dir / "latent_v" / f"{axis}_{epoch}.pickle",
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
            if model_dir:
                save_samples(samples, f"{axis}_{epoch}", model_dir / "Audio")

            ax = plt.figure(figsize=(5, 5)).add_subplot(projection='3d')
            #ax.invert_yaxis()

            ax.view_init(30, 30)
            start, end = filter_start * len(samples[0][0]), filter_end * len(
                samples[0][0])
            print(start, end)
            start, end = int(start), int(end)
            sec_per_sample = 1.0 / 16000
            x = np.arange(0, sec_per_sample * len(samples[0, 0, start:end]),
                          sec_per_sample)
            print(x.shape, samples[:, :, start:end].shape)
            print()
            for i in range(batch_size):
                print(f"Plot {i}: {alter_vals[i]}")
                ax.plot(
                    x,
                    samples[i, 0, start:end],
                    zs=alter_vals[i],
                    zdir="x",
                    label=f"{i}",
                )
            ax.set_xlabel(f'L {axis}')
            ax.set_xlim(xmin=alter_start, xmax=alter_end)
            ax.set_ylabel('Time')
            ax.set_zlabel('Amplitude')

            print('ax.azim {}'.format(ax.azim))

            print('ax.elev {}'.format(ax.elev))

            plt.savefig(model_dir / "Audio" / f"{axis}_{epoch}.png", dpi=600)
            plt.close()
            with open(model_dir / "Audio" / f"{axis}_{epoch}.txt",
                      'w') as fout:
                np.savetxt(fname=fout, X=alter_vals)
