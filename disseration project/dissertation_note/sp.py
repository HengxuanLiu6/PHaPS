import torch
import torchaudio
import os

def speed_perturb_dataset(
        dataset,
        output_dir="augmented_audio_PHaPS_SP",
        factors=(0.9, 1.1)       # < 1 ＝放慢；> 1 ＝加快
):
    os.makedirs(output_dir, exist_ok=True)

    def apply_speed_perturbation(batch):
        audio      = batch["audio"]
        waveform   = torch.tensor(audio["array"]).unsqueeze(0).float()  # [1, T]
        orig_sr    = audio["sampling_rate"]                             # 16 000 for Whisper
        base_name  = os.path.splitext(os.path.basename(audio["path"]))[0]

        results = {}

        for f in factors:
            # 核心：new_sr = orig_sr / f   （只 resample 一次）
            new_sr   = int(orig_sr / f)
            suffix   = f"sp{str(f).replace('.', '')}"
            resampled = torchaudio.functional.resample(
                waveform, orig_sr, new_sr
            )

            # ⬇ 保存时仍然标注 16 kHz，所以播放时速度=1/f
            out_path = os.path.join(output_dir, f"{base_name}_{suffix}.wav")
            torchaudio.save(out_path, resampled, orig_sr)

            results[f"audio_{suffix}"] = out_path

        results["sentence"] = batch["sentence"]
        return results

    return dataset.map(apply_speed_perturbation)
