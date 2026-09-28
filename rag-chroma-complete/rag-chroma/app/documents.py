"""Page-aware extraction and overlapping CHARACTER chunks (not tokens)."""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import PurePosixPath
import re
import unicodedata
from pypdf import PdfReader

@dataclass(frozen=True)
class Chunk:
    text: str
    page: int  # 1-based PDF page; 0 means a text/Markdown file
    index: int


def source_name(filename: str) -> str:
    # Source identity is a basename; repeated names intentionally replace.
    name = PurePosixPath(filename.replace('\\', '/')).name
    if not name or len(name) > 160 or re.search(r'[\x00-\x1f\x7f]', name):
        raise ValueError('Invalid filename')
    if PurePosixPath(name).suffix.lower() not in {'.pdf', '.txt', '.md'}:
        raise ValueError('Supported files: PDF, TXT, MD')
    return name


def clean(text: str) -> str:
    text = unicodedata.normalize('NFC', text).replace('\x00', '')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[^\S\n]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def extract(raw: bytes, source: str) -> list[tuple[int, str]]:
    if source.lower().endswith('.pdf'):
        try:
            reader = PdfReader(BytesIO(raw))
            if reader.is_encrypted:
                raise ValueError('Encrypted PDFs are not supported; upload an unlocked copy')
            if len(reader.pages) > 500:
                raise ValueError('Maximum 500 PDF pages per upload')
            pages = []
            total = 0
            for i, page in enumerate(reader.pages, 1):
                text = clean(page.extract_text() or '')
                total += len(text)
                if total > 2_000_000:
                    raise ValueError('Maximum 2 million extracted characters per file')
                pages.append((i, text))
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError('PDF could not be read') from exc
    else:
        try:
            pages = [(0, clean(raw.decode('utf-8-sig')))]
        except UnicodeDecodeError as exc:
            raise ValueError('TXT and MD files must use UTF-8 encoding') from exc
    if sum(len(t) for _, t in pages) > 2_000_000:
        raise ValueError('Maximum 2 million extracted characters per file')
    if not any(text for _, text in pages):
        raise ValueError('No extractable text. Scanned PDFs need OCR first')
    return pages


def split_pages(pages: list[tuple[int, str]], size: int, overlap: int) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError('Require 0 <= overlap < size')
    chunks = []
    for page, text in pages:
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                # Prefer a paragraph, line, or word boundary in the last half.
                floor = start + max(size // 2, overlap + 1)
                for separator in ('\n\n', '\n', ' '):
                    boundary = text.rfind(separator, floor, end)
                    if boundary != -1:
                        end = boundary + len(separator)
                        break
            part = text[start:end].strip()
            if part:
                chunks.append(Chunk(part, page, len(chunks)))
            if end == len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks


def digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()
