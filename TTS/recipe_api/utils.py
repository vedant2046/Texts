import os

from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.shared_configs import BaseTTSConfig, CharactersConfig, GSTConfig


def gst_config(size="small"):
    if size == "small":
        gst = GSTConfig(gst_num_heads=4, gst_num_style_tokens=5, gst_embedding_dim=128)
    elif size == "large":
        gst = GSTConfig(gst_num_heads=5, gst_num_style_tokens=10, gst_embedding_dim=512)
    return gst


def data_loader(name, path, stats_path=None):
    if name == "ljspeech":
        dataset = BaseDatasetConfig(name="ljspeech", meta_file_train="metadata.csv", path=path)
        audio = BaseAudioConfig(
            ref_level_db=0, trim_db=60, mel_fmin=50.0, mel_fmax=7600.0, spec_gain=1, stats_path=stats_path
        )
    elif name == "vctk":
        dataset = BaseDatasetConfig(
            name="vctk",
            meta_file_train=["p225", "p234", "p238", "p245", "p248", "p261", "p294", "p302", "p326", "p335", "p347"],
            meta_file_val=None,
            path=path,
        )
        audio = BaseAudioConfig(
            sample_rate=22050,
            preemphasis=0.98,
            ref_level_db=20,
            clip_norm=True,
            mel_fmin=0.0,
            mel_fmax=8000.0,
            spec_gain=20,
            do_trim_silence=False,
            trim_db=60,
            power=1.5,
            num_mels=80,
            resample=True,
        )

    elif name == "libri_tts":
        dataset = BaseDatasetConfig(name="libri_tts", meta_file_train=None, meta_file_val=None, path=path)
        audio = BaseAudioConfig(
            resample=False,
            sample_rate=24000,
            preemphasis=0.98,
            ref_level_db=20,
            power=1.5,
            signal_norm=True,
            symmetric_norm=True,
            max_norm=4.0,
            clip_norm=True,
            mel_fmax=8000.0,
            spec_gain=20,
            do_trim_silence=False,
            trim_db=25,
        )
    elif name == "sam" or "sam_accenture":
        dataset = BaseDatasetConfig(name=name, meta_file_train="recording_script.xml", meta_file_val=None, path=path)
        audio = BaseAudioConfig(
            sample_rate=16000,
            preemphasis=0.0,
            signal_norm=False,
            clip_norm=True,
            mel_fmax=8000.0,
            spec_gain=1,
            do_trim_silence=True,
            trim_db=60,
            symmetric_norm=True,
            num_mels=80,
        )
    elif name == "baker":
        dataset = BaseDatasetConfig(name=name, meta_file_train="metadata.csv", meta_file_val=None, path=path)
        audio = BaseAudioConfig(
            sample_rate=22050,
            preemphasis=0.0,
            ref_level_db=0,
            do_trim_silence=True,
            trim_db=60,
            mel_fmin=50.0,
            mel_fmax=7600.0,
            spec_gain=1,
            signal_norm=True,
            symmetric_norm=True,
            clip_norm=True,
            stats_path=stats_path,
        )
    return dataset, audio
