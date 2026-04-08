import re
import logging
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def clean_markdown(text: str) -> str:
    text = re.sub(r"\A---\n.*?\n---(?:\n|$)", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"!\[\[.*?\]\]", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_long_section(title: str, content: str, max_chars: int = 1200) -> list[dict]:
    paragraphs = content.split("\n\n")
    chunks = []
    current_text = ""

    for p in paragraphs:
        if len(current_text) + len(p) < max_chars:
            current_text += p + "\n\n"
        else:
            if current_text.strip():
                chunks.append({"title": title, "content": current_text.strip()})
            current_text = p + "\n\n"

    if current_text.strip():
        chunks.append({"title": title, "content": current_text.strip()})

    return chunks

def chunk_by_heading(title: str, content: str, min_length: int = 50, max_length: int = 1200) -> list[dict]:
    content = clean_markdown(content)
    sections = re.split(r"(?:^|\n)(?=#{1,3} )", content)
    chunks = list()

    for section in sections:
        section = section.strip()
        if not section:
            continue

        lines = section.splitlines()
        heading = ""
        if lines[0].startswith("#"):
            heading = lines[0].lstrip("#").strip()

        chunk_title = f"{title} > {heading}" if heading else title

        if len(section) > max_length:
            sub_chunks = split_long_section(chunk_title, section, max_length)
            chunks.extend(sub_chunks)
        elif len(section) >= min_length:
            chunks.append({"title": chunk_title, "content": section})

    return chunks

def read_vault(
        vault_path: str,
        exclude_folders: list[str] | None = None,
        exclude_files: list[str] | None = None,
) -> list[dict]:
    exclude_folders = set(exclude_folders or ["templates", ".trash", ".obsidian"])
    exclude_files = set(exclude_files or [])

    vault = Path(vault_path)
    records = []

    for md_file in vault.rglob("*.md"):
        if any(part in exclude_folders for part in md_file.parts):
            continue

        if md_file.name in exclude_files:
            continue

        try:
            raw = md_file.read_text(encoding="utf-8")
            title = md_file.stem
            rel = str(md_file.relative_to(vault))

            content = clean_markdown(raw)

            if content:
                records.append({
                    "title": title,
                    "content": content,
                    "source": rel,
                })

        except Exception as e:
            log.warning(f"Could not read {md_file}: {e}")

    log.info(f"Loaded {len(records)} notes from {vault_path}")
    return records

def build_knowledge_base(
        vault_path: str,
        output_path: str = "knowledge_base.pkl",
        model_name: str = "BAAI/bge-small-en-v1.5",
        exclude_folders: list[str] | None = None,
        exclude_files: list[str] | None = None
) -> pd.DataFrame:
    records = read_vault(vault_path, exclude_folders, exclude_files)

    if not records:
        raise ValueError(f"No markdown files found in: {vault_path}")

    log.info(f"Loading embedding model: {model_name}")
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

    model = SentenceTransformer(model_name, trust_remote_code=True)
    df = pd.DataFrame(records)
    contents = ("Document: " + df["title"] + "\n\n" + df["content"]).tolist()

    embeddings = model.encode(
        contents,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64
    )

    df["embedding"] = list(embeddings)

    df.to_pickle(output_path)
    log.info(f"Saved knowledge base: {output_path}  ({len(df)} chunks)")

    return df