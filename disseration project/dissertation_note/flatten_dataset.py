from pathlib import Path
from datasets import Dataset
import torchaudio
import numpy as np
from typing import Dict, List, Union

def flatten_dataset(
    fold_dataset: Dataset,
    variant_map: Dict[str, Union[str, List[str]]] = None,
    keep_original: bool = True,
    target_dtype=np.float32,
):
    """
    将带有多种增强版本的条目“摊平”为一行一条音频。

    Parameters
    ----------
    fold_dataset : Dataset
        每条记录包含
            ├─ audio : {'array': ndarray, 'sampling_rate': int}
            ├─ sentence : str
            └─ 若干增强列（如 'audio_sp09', 'audio_pu2' …）
    variant_map : dict
        键 = 想写进结果的 source 字段
        值 = 对应 fold_dataset 里的列名(单个或列表)
        例如：
            {
              'sp09': 'audio_sp09',
              'sp11': 'audio_sp11',
              'pp-2': 'audio_pn2',
              'pp+2': ['audio_pu2', 'audio_pu3']  # 也可以给列表
            }
    keep_original : bool
        是否把原始音频也写进去（source='original'）
    target_dtype : np.dtype
        波形数组输出的数据类型
    """
    new_rows = []

    # ----------- 遍历原 Dataset 行 -----------
    for item in fold_dataset:
        # 1) 原始
        if keep_original:
            new_rows.append({
                'audio_path': None,
                'audio_array': np.asarray(item['audio']['array'], dtype=target_dtype),
                'sampling_rate': item['audio']['sampling_rate'],
                'sentence': item['sentence'],
                'source': 'original',
            })

        # 2) 各种增强
        for src_tag, col_names in (variant_map or {}).items():
            # 允许 value 是字符串或列表
            if isinstance(col_names, str):
                col_names = [col_names]

            for col in col_names:
                wav_path = item[col]                           # 文件路径
                waveform, sr = torchaudio.load(wav_path)       # [1, T]
                new_rows.append({
                    'audio_path': wav_path,
                    'audio_array': waveform.squeeze().numpy().astype(target_dtype),
                    'sampling_rate': sr,
                    'sentence': item['sentence'],
                    'source': src_tag,
                })

    return Dataset.from_list(new_rows)