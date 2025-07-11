import os, numpy as np, soundfile as sf, librosa
from pathlib import Path

def pitch_perturb_dataset(
        dataset,
        output_dir="aug_pitch",
        n_steps_list=(-2, +2)   # 半音；可以再加 ±3
):
    os.makedirs(output_dir, exist_ok=True)

    def augment(batch):
        y  = batch["audio"]["array"]          # 1‑D numpy.float32
        sr = batch["audio"]["sampling_rate"]  # 应该是 16000

        # 生成用于保存文件名的“基名”
        # 如果确实有 path 就用它的 stem；否则用 dataset id
        base_name = (Path(batch["audio"].get("path", "sample")).stem
                     if isinstance(batch["audio"], dict)
                     else f"sample_{batch['__index_level_0__']}")

        results = {}
        for n in n_steps_list:
            # ---------- 1. pitch‑shift ----------
            y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=n)

            # ---------- 2. 保存 ----------
            suffix  = f"p{n:+d}".replace("+", "u")   # +2 → pu2
            outpath = os.path.join(output_dir, f"{base_name}_{suffix}.wav")
            sf.write(outpath, y_shift, sr)           # 仍是 16 kHz

            results[f"audio_{suffix}"] = outpath

        results["sentence"] = batch["sentence"]
        return results

    return dataset.map(augment)

