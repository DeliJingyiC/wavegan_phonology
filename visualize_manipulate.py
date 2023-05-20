import argparse
import itertools
import os
from pathlib import Path
import torch
from tqdm import trange
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
import soundfile as sf
from matplotlib.gridspec import GridSpec


def parse_arguments():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Analyze a fiwGAN on a given latent code c')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--input_dir', type=str)

    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_arguments()

    input_dir = Path(args["input_dir"])
    output_dir = Path(args["output_dir"])

    laten_v_dir = input_dir / "latent_v"
    audio_dir = input_dir / "Audio"

    with open(input_dir / f"params.json", 'r') as fin:
        source_job_params = json.load(fin)

    alter_axis = [int(x) for x in source_job_params["alter_axis"].split(",")]

    assert len(alter_axis) == 2, f"Cannot visualize more than 2 alter axis"

    pkl_files = sorted(list(laten_v_dir.iterdir()))

    pkl_loaded = []

    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as fin:
            pkl_loaded.append(pickle.load(fin))

    audio_files = sorted(
        [x for x in audio_dir.iterdir() if x.name.endswith(".wav")])

    audio_loaded = []

    for audio_path in audio_files:

        audio, sr = sf.read(audio_path)

        audio_loaded.append(audio)

    print(len(pkl_loaded), pkl_loaded[0].shape)
    print(len(audio_loaded), audio_loaded[0].shape)

    laten_v = np.vstack(pkl_loaded)
    audio = np.vstack(audio_loaded)

    assert len(laten_v) == len(audio), f"{len(laten_v)}=={len(audio)}"

    latent_v: np.ndarray = laten_v[:, alter_axis]
    alter_start, alter_end = min(np.unique(latent_v[:, 1])), max(
        np.unique(latent_v[:, 1])),
    print(latent_v)

    print(latent_v.shape)
    print(audio.shape)

    x_mesh, y_mesh = np.meshgrid(np.unique(latent_v[:, 0]),
                                 np.unique(latent_v[:, 1]))

    print(x_mesh.shape)
    print(y_mesh.shape)

    assert len(latent_v) == np.product(
        x_mesh.shape
    ), f"Latent_v metrix is missing part of it. {len(latent_v)}/{np.product(x_mesh.shape)}"

    audio = audio.reshape(*x_mesh.shape, audio.shape[-1])

    print(audio.shape)
    fig = plt.figure(figsize=(40, 5), dpi=600)
    gs = GridSpec(1, audio.shape[0], figure=fig)
    """
    fig, axes = plt.subplots(
        nrows=1,
        ncols=audio.shape[0],
        sharex="all",
        sharey="row",
        squeeze=True,
        subplot_kw={"projection": "3d"},
    )
    """

    sec_per_sample = 1.0 / 16000
    filter_start, filter_end = [
        float(x) for x in source_job_params["filter_range"].split(",")
    ]
    start, end = filter_start * len(audio[0][0]), filter_end * len(audio[0][0])

    start, end = int(start), int(end)
    x = np.arange(0, sec_per_sample * (end - start), sec_per_sample)
    for i in range(audio.shape[0]):

        axis = fig.add_subplot(gs[0, i:i + 1], projection="3d")
        axis.view_init(30, 30)

        for j in range(audio.shape[1]):
            axis.plot(x,
                      audio[i][j][start:end],
                      zs=latent_v[i * audio.shape[1] + j][-1],
                      zdir="x")
        axis.set_xlabel(f'L {alter_axis[0]}={latent_v[i * audio.shape[1]][0]}')
        axis.set_xlim(xmin=alter_start, xmax=alter_end)
        axis.set_ylabel('Time')
        axis.set_zlabel('Amplitude')
    """
    plt.subplots_adjust(
        #left=0.125,
        #bottom=0.1,
        #right=0.9,
        top=0.9,
        wspace=0.12,
        hspace=0.09,
    )
    """
    plt.savefig(output_dir / "view.png")
