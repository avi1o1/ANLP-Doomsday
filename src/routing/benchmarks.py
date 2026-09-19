"""Prepare WikiText, MMLU and GSM8K for expert-routing evaluation."""

from __future__ import annotations

from pathlib import Path

from src.artifacts import Artifact, atomic_json, write_jsonl


def mmlu_prompt(record):
    choices = "\n".join(f"{chr(65+i)}. {text}" for i, text in enumerate(record["choices"]))
    return f"{record['question']}\n{choices}\nAnswer:"


def prepare_routing(config, benchmark, stop):
    from datasets import load_dataset

    spec = config["routing"]
    directory = Path(spec["data"]).resolve() / benchmark
    source = {"wikitext": ("Salesforce/wikitext", "wikitext-2-raw-v1"),
              "mmlu": ("cais/mmlu", "all"), "gsm8k": ("openai/gsm8k", "main")}[benchmark]
    revision = spec.get("dataset_revisions", {}).get(benchmark, "main")
    with Artifact(directory, {"source": source, "revision": revision, "protocol": "zero-shot-fixed-ids-v1"}) as work:
        if work.reused:
            return directory
        dataset = load_dataset(*source, revision=revision)
        metadata = {"source": source, "revision": revision,
                    "fingerprints": {split: data._fingerprint for split, data in dataset.items()}}
        if benchmark == "wikitext":
            for split in ("train", "test"):
                write_jsonl(directory / f"{split}.jsonl",
                            ({"id": str(i), "text": r["text"]} for i, r in enumerate(dataset[split])))
            outputs = ["train.jsonl", "test.jsonl"]
        elif benchmark == "mmlu":
            records = []
            for i, record in enumerate(dataset["test"]):
                stop.check()
                records.append({"id": f"{record['subject']}:{i}", "subject": record["subject"],
                                "prompt": mmlu_prompt(record), "answer": int(record["answer"]),
                                "choices": [" A", " B", " C", " D"]})
            write_jsonl(directory / "test.jsonl", records)
            outputs = ["test.jsonl"]
        else:
            for split in ("train", "test"):
                write_jsonl(directory / f"{split}.jsonl", ({"id": str(i),
                    "prompt": r["question"] + "\nAnswer:", "answer": r["answer"].split("####")[-1].strip()}
                    for i, r in enumerate(dataset[split])))
            outputs = ["train.jsonl", "test.jsonl"]
        atomic_json(directory / "source.json", metadata)
        work.complete(outputs + ["source.json"])
    return directory
