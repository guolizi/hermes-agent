"""Media utility functions: thumbnail generation and orphan cleanup.

All operations are best-effort (failures are logged, never crash).
"""
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Thumbnail generation ──────────────────────────────────────────

# Max dimensions for thumbnails (keep aspect ratio)
_THUMB_MAX_W = 320
_THUMB_MAX_H = 240
_THUMB_QUALITY = 75

# Files larger than this get a thumbnail
_THUMB_MIN_BYTES = 50 * 1024  # 50 KB


def generate_thumbnail(
    source_path: str,
    mime_type: str,
    media_root: str,
) -> Optional[str]:
    """Generate a thumbnail for an image file, stored alongside the original.

    Args:
        source_path: Absolute path to the original media file.
        mime_type: MIME type (e.g. 'image/jpeg').
        media_root: Root media directory (e.g. /path/to/media).

    Returns:
        Relative path to the thumbnail (e.g. 'thumbs/im/ca/cafe.jpg'),
        or None if thumbnail not applicable / failed.
    """
    if not mime_type.startswith("image/"):
        return None

    try:
        from PIL import Image, ImageOps
    except ImportError:
        logger.debug("Pillow not available, skipping thumbnail for %s", source_path)
        return None

    # Skip non-image mime types within image/* (e.g. image/svg+xml is vector)
    ext = source_path.rsplit(".", 1)[-1].lower() if "." in source_path else ""
    if ext in ("svg", "svgz") or mime_type in ("image/svg+xml", "image/svg"):
        return None

    # Check file size — skip if already small
    try:
        fsize = os.path.getsize(source_path)
    except OSError:
        return None
    if fsize < _THUMB_MIN_BYTES:
        return None

    # Compute thumbnail path: thumbs/{type_code}/{sha[:2]}/{filename}
    rel_path = _thumbnail_rel_path(source_path, media_root)
    if rel_path is None:
        return None

    # Force .jpg extension for thumbnail (saved as JPEG regardless of source)
    base, _ = os.path.splitext(rel_path)
    rel_path = base + ".jpg"
    thumb_abs = Path(media_root) / rel_path
    if thumb_abs.exists():
        return rel_path  # already generated

    try:
        thumb_abs.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(source_path)
        img = ImageOps.exif_transpose(img) or img  # auto-orient
        img.thumbnail((_THUMB_MAX_W, _THUMB_MAX_H), Image.LANCZOS)
        # Convert RGBA/P to RGB for JPEG
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        img.save(str(thumb_abs), "JPEG", quality=_THUMB_QUALITY)
        logger.info("Thumbnail generated: %s (%dx%d, %d bytes)",
                     rel_path, img.width, img.height,
                     thumb_abs.stat().st_size)
        return rel_path
    except Exception as exc:
        logger.warning("Thumbnail generation failed for %s: %s", source_path, exc)
        # Remove partial file if any
        thumb_abs.unlink(missing_ok=True)
        return None


def _thumbnail_rel_path(source_path: str, media_root: str) -> Optional[str]:
    """Compute the relative thumbnail path from an absolute media path.

    E.g. /path/media/im/ca/cafe.jpg → thumbs/im/ca/cafe.jpg
    """
    try:
        src = Path(source_path).resolve()
        root = Path(media_root).resolve()
        rel = src.relative_to(root)
        return str(Path("thumbs") / rel)
    except (ValueError, RuntimeError):
        # source_path not under media_root — can't create thumbnail
        return None


# ── Orphan cleanup ────────────────────────────────────────────────

def media_cleanup(store, dry_run: bool = True) -> dict:
    """Remove orphaned media files from disk.

    Args:
        store: MemoryStore instance.
        dry_run: If True, only report orphans without deleting.

    Returns:
        Dict with keys: deleted (int), skipped (int), freed_bytes (int),
        dry_run (bool), errors (list[str]).
    """
    orphans = store.media_orphans()
    if not orphans:
        return {"deleted": 0, "skipped": 0, "freed_bytes": 0,
                "dry_run": dry_run, "errors": []}

    media_root = Path(store._media_dir)
    deleted = 0
    protected = 0
    not_found = 0
    freed_bytes = 0
    errors = []

    for rel in orphans:
        fpath = media_root / rel
        if not fpath.exists():
            not_found += 1
            continue

        # Protect thumbnails — only delete if the original is also orphaned
        if rel.startswith("thumbs/"):
            # Check if original file is also an orphan
            # Thumbnail saves as .jpg regardless of original extension
            orig_rel = rel.replace("thumbs/", "", 1)
            # Try with common image extensions (thumbnail always .jpg)
            orig_no_ext = os.path.splitext(orig_rel)[0]
            orig_found = any(
                o in orphans
                for o in [orig_rel, orig_no_ext + ".png",
                          orig_no_ext + ".jpg", orig_no_ext + ".jpeg",
                          orig_no_ext + ".gif", orig_no_ext + ".webp",
                          orig_no_ext + ".tiff", orig_no_ext + ".tif",
                          orig_no_ext + ".bmp", orig_no_ext + ".heic",
                          orig_no_ext + ".heif", orig_no_ext + ".avif",
                          orig_no_ext + ".ico"]
            )
            if not orig_found:
                protected += 1
                continue  # original exists, keep thumbnail

        try:
            fsize = fpath.stat().st_size
            if dry_run:
                logger.debug("GC would delete orphan: %s (%d bytes)", rel, fsize)
            else:
                fpath.unlink()
                logger.info("GC deleted orphan: %s (%d bytes)", rel, fsize)
                deleted += 1
                freed_bytes += fsize
        except OSError as e:
            errors.append(str(e))
            logger.warning("GC failed to delete %s: %s", rel, e)

    # Remove empty directories (reverse order to handle nesting)
    if not dry_run:
        for dirpath, dirnames, filenames in os.walk(str(media_root), topdown=False):
            if not dirnames and not filenames:
                try:
                    os.rmdir(dirpath)
                except OSError:
                    pass

    return {
        "deleted": deleted,
        "protected": protected,
        "not_found": not_found,
        "freed_bytes": freed_bytes,
        "dry_run": dry_run,
        "errors": errors,
    }
