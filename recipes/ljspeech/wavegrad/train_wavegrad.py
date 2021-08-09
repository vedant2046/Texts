import os

from TTS.trainer import Trainer, TrainingArgs, init_training
from TTS.vocoder.configs import WavegradConfig

output_path = os.path.dirname(os.path.abspath(__file__))
config = WavegradConfig(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    seq_len=6144,
    pad_short=2000,
    use_noise_augment=True,
    eval_split_size=50,
    print_step=50,
    print_eval=True,
    mixed_precision=False,
    data_path=os.path.join(output_path, "../LJSpeech-1.1/wavs/"),
    output_path=output_path,
)
args, config, output_path, _, c_logger, dashboard_logger = init_training(TrainingArgs(), config)
trainer = Trainer(args, config, output_path, c_logger, dashboard_logger)
trainer.fit()
