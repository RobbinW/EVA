from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm import trange

from datasets.robotwin import RobotwinDataset
from .exp_base import BaseExperiment


class WanPromptEncoder:
    def __init__(self, cfg):
        from algorithms.wan.modules.t5 import umt5_xxl
        from algorithms.wan.modules.tokenizers import HuggingfaceTokenizer

        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.text_encoder = (
            umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=self.dtype,
                device=torch.device("cpu"),
            )
            .eval()
            .requires_grad_(False)
        )

        ckpt_path = cfg.text_encoder.get("ckpt_path", None)
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self.text_encoder.load_state_dict(state_dict)
        else:
            raise ValueError("algorithm.text_encoder.ckpt_path is required for cache_prompt_embed.")

        self.text_encoder = self.text_encoder.to(self.device)
        self.tokenizer = HuggingfaceTokenizer(
            name=cfg.text_encoder.name,
            seq_len=cfg.text_encoder.text_len,
            clean="whitespace",
        )

    @torch.no_grad()
    def predict(self, prompts: list[str]) -> list[torch.Tensor]:
        ids, mask = self.tokenizer(prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        return [u[:v].cpu() for u, v in zip(context, seq_lens)]


def build_prompt_encoder(root_cfg):
    algo_name = root_cfg.algorithm._name
    if "wan" not in algo_name:
        raise ValueError(
            f"cache_prompt_embed only supports Wan algorithms, got {algo_name}."
        )
    return WanPromptEncoder(root_cfg.algorithm)


def _relative_to_root(path_like: str | Path, root_dir: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(root_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Absolute path {path} must be inside dataset root {root_dir}."
            ) from exc
    return path


def _resolve_prompt_embed_relpath(record: dict, data_root: Path) -> Path:
    existing = str(record.get("prompt_embed_path", "") or "").strip()
    if existing:
        return _relative_to_root(existing, data_root)
    return _relative_to_root(record["video_path"], data_root).with_suffix(".pt")


def _resolve_metadata_relpath(metadata_path: Path, data_root: Path) -> Path:
    if metadata_path.is_absolute():
        try:
            return metadata_path.resolve().relative_to(data_root.resolve())
        except ValueError:
            return Path(metadata_path.name)
    return metadata_path


class ProcessDataExperiment(BaseExperiment):
    compatible_datasets = dict(
        robotwin=RobotwinDataset,
    )

    def _build_dataset(self, disable_filtering: bool = True, split: str = "all"):
        dataset_name = self.root_cfg.dataset._name
        if dataset_name not in self.compatible_datasets:
            raise ValueError(
                f"Dataset {dataset_name} is not registered for ProcessDataExperiment."
            )
        if disable_filtering:
            self.root_cfg.dataset.filtering.disable = True
        return self.compatible_datasets[dataset_name](self.root_cfg.dataset, split=split)

    def _get_save_dir(self, dataset) -> Path:
        save_dir = self.cfg.new_data_root
        if save_dir is None:
            save_dir = self.output_dir / dataset.data_root.name
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        return Path(save_dir)

    def cache_prompt_embed(self):
        cfg = self.cfg.cache_prompt_embed
        if self.cfg.num_nodes != 1:
            raise ValueError("cache_prompt_embed only supports num_nodes=1.")

        prompt_encoder = build_prompt_encoder(self.root_cfg)
        dataset = self._build_dataset()
        records = dataset.records

        save_dir = self._get_save_dir(dataset)
        metadata_relpath = _resolve_metadata_relpath(dataset.metadata_path, dataset.data_root)
        metadata_path = save_dir / metadata_relpath
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        new_records = []
        batch_size = cfg.batch_size
        for start in trange(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            prompts = [dataset.id_token + r.get("caption", "") for r in batch]
            embeds = prompt_encoder.predict(prompts)

            for record, embed in zip(batch, embeds):
                prompt_embed_relpath = _resolve_prompt_embed_relpath(
                    record, dataset.data_root
                )
                prompt_embed_path = save_dir / prompt_embed_relpath
                prompt_embed_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(embed.clone(), prompt_embed_path)

                updated = dict(record)
                updated["prompt_embed_path"] = str(prompt_embed_relpath)
                new_records.append(updated)

        pd.DataFrame.from_records(new_records).to_csv(metadata_path, index=False)

        print(f"Saved prompt embeddings and metadata to {save_dir}")
