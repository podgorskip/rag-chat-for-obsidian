import re
import logging
import pandas as pd
from pathlib import Path
from embedders.embedder import Embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def clean_markdown(text: str) -> str:
    text = re.sub(r"\A---\n.*?\n---(?:\n|$)", "", text, flags=re.DOTALL)    # front-matter
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)              # aliased wikilinks
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)                         # plain wikilinks
    text = re.sub(r"!\[\[.*?\]\]", "", text)                                # embedded images
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)                             # markdown images
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)                   # hyperlinks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)                  # fenced code blocks
    text = re.sub(r"`[^`]+`", "", text)                                     # inline code
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])|(?<=\n)", text)
    return [p.strip() for p in parts if p.strip()]

def _build_chunks_with_overlap(sentences: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0

    while start < len(sentences):
        current, length = [], 0
        i = start

        while i < len(sentences):
            s = sentences[i]
            if length + len(s) + 1 > chunk_size and current:
                break
            current.append(s)
            length += len(s) + 1
            i += 1

        if not current:
            current = [sentences[start]]
            i = start + 1

        chunks.append(" ".join(current))

        back, acc = i - 1, 0
        while back > start and acc < overlap:
            acc += len(sentences[back])
            back -= 1
        start = max(back + 1, start + 1)

    return chunks

def chunk_document(
        title: str,
        content: str,
        chunk_size: int = 1_000,
        overlap: int = 150,
        min_length: int = 60,
) -> list[dict]:

    content = clean_markdown(content)
    raw_sections = re.split(r"(?m)(?=^#{1,3} )", content)
    chunks: list[dict] = []

    for section in raw_sections:
        section = section.strip()
        if not section:
            continue

        lines = section.splitlines()
        if lines[0].startswith("#"):
            heading = lines[0].lstrip("#").strip()
            body = "\n".join(lines[1:]).strip()
        else:
            heading = ""
            body = section

        section_title = f"{title} > {heading}" if heading else title

        if not body:
            continue

        sentences = _split_sentences(body)
        raw_chunks = _build_chunks_with_overlap(sentences, chunk_size, overlap)

        for chunk_text in raw_chunks:
            if len(chunk_text) < min_length:
                if chunks:
                    chunks[-1]["content"] += " " + chunk_text
                continue
            chunks.append({"title": section_title, "content": chunk_text})

    return chunks

def read_vault(
        vault_path: str,
        exclude_folders: list[str] | None = None,
        exclude_files: list[str] | None = None,
        chunk_size: int = 1_000,
        overlap: int = 150,
        min_length: int = 60,
) -> list[dict]:
    exclude_folders = set(exclude_folders or ["templates", ".trash", ".obsidian"])
    exclude_files = set(exclude_files or [])

    vault = Path(vault_path)
    records: list[dict] = []

    for md_file in vault.rglob("*.md"):
        if any(part in exclude_folders for part in md_file.parts):
            continue
        if md_file.name in exclude_files:
            continue

        try:
            raw = md_file.read_text(encoding="utf-8")
            title = md_file.stem
            rel = str(md_file.relative_to(vault))

            for chunk in chunk_document(title, raw, chunk_size, overlap, min_length):
                records.append({**chunk, "source": rel})

        except Exception as e:
            log.warning(f"Could not read {md_file}: {e}")

    log.info(f"Produced {len(records)} chunks from vault: {vault_path}")
    return records

def build_knowledge_base(
        vault_path: str,
        embedder: Embedder,
        output_path: str = "knowledge_base.pkl",
        exclude_folders: list[str] | None = None,
        exclude_files: list[str] | None = None,
        chunk_size: int = 1_000,
        overlap: int = 150,
) -> pd.DataFrame:
    records = read_vault(vault_path, exclude_folders, exclude_files, chunk_size, overlap)

    if not records:
        raise ValueError(f"No markdown files found in: {vault_path}")

    df = pd.DataFrame(records)
    texts = ("Document: " + df["title"] + "\n\n" + df["content"]).tolist()
    embeddings = embedder.encode(texts)

    df["embedding"] = list(embeddings)
    df.to_pickle(output_path)
    log.info(f"Saved knowledge base → {output_path}  ({len(df)} chunks)")

    return df