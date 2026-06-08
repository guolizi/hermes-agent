"""Media compression utilities: image (Pillow), video/audio (ffmpeg).

All operations are best-effort (failures are logged, never crash).
If compression fails or doesn't save space, the original file is kept.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Default compression config ──────────────────────────────────────

DEFAULT_COMPRESSION_CONFIG = {
    "enabled": True,
    "max_size_mb": 100,           # skip compression for files larger than this
    "timeout": 600,               # ffmpeg timeout in seconds (10 min default)
    "image": {
        "quality": 85,               # JPEG quality 1-100
        "max_dim": 1920,             # resize if width/height > this
        "convert_to_jpeg": True,     # PNG/GIF → JPEG (lossy, much smaller)
    },
    "video": {
        "bitrate": "1M",             # ffmpeg -b:v
        "max_fps": 30,
        "max_dim": 1280,
        "audio_bitrate": "128k",     # ffmpeg -b:a
    },
    "audio": {
        "bitrate": "128k",           # ffmpeg -b:a
        "sample_rate": 44100,
    },
}

_IMAGE_SAFE_EXT = {"jpg", "jpeg", "png", "gif", "webp", "tiff", "tif",
                    "bmp", "ico", "heic", "heif", "avif"}

# Cache for ffmpeg availability check
_FFMPEG_AVAILABLE: Optional[bool] = None


def compress_media(source_path: str, mime_type: str,
                   output_dir: str,
                   config: Optional[dict] = None) -> tuple[Optional[str], Optional[str]]:
    """Compress a media file if beneficial.

    Args:
        source_path: Absolute path to the original file.
        mime_type: MIME type (e.g. 'image/png').
        output_dir: Directory to write the compressed file into (same filesystem).
        config: Compression config dict (partial overrides of DEFAULT_COMPRESSION_CONFIG).

    Returns:
        (compressed_path, updated_mime_type) or (None, None) if compression skipped.
        If compression succeeded, the caller should use compressed_path instead of
        source_path and updated_mime_type instead of mime_type.
    """
    if config is None:
        config = DEFAULT_COMPRESSION_CONFIG
    if not config.get("enabled", True):
        return None, None

    # Skip compression for files exceeding max_size_mb
    max_size_mb_val = int(config.get("max_size_mb", 100))
    if max_size_mb_val <= 0:
        max_size_mb_val = 100
    max_size_bytes = max_size_mb_val * 1024 * 1024
    try:
        file_size = os.path.getsize(source_path)
    except OSError:
        file_size = 0
    if file_size > max_size_bytes:
        logger.warning(
            "Compression skipped for %s: %d bytes exceeds max_size=%dMB",
            source_path, file_size, max_size_bytes // (1024 * 1024),
        )
        return None, None

    # Determine timeout for ffmpeg operations
    _timeout = int(config.get("timeout", 600))

    try:
        if mime_type.startswith("image/"):
            img_cfg = {**DEFAULT_COMPRESSION_CONFIG["image"], **config.get("image", {})}
            return _compress_image(source_path, img_cfg)
        elif mime_type.startswith("video/"):
            vid_cfg = {**DEFAULT_COMPRESSION_CONFIG["video"], **config.get("video", {})}
            return _compress_video(source_path, output_dir, vid_cfg, _timeout)
        elif mime_type.startswith("audio/"):
            aud_cfg = {**DEFAULT_COMPRESSION_CONFIG["audio"], **config.get("audio", {})}
            return _compress_audio(source_path, output_dir, aud_cfg, _timeout)
    except Exception:
        logger.warning("Compression failed for %s (type=%s)", source_path, mime_type,
                       exc_info=True)

    return None, None


def _compress_image(source_path: str, img_cfg: dict) -> tuple[Optional[str], Optional[str]]:
    """Compress an image using Pillow.

    Returns (compressed_path, "image/jpeg") or (None, None).
    JPEG output is guaranteed regardless of input format for max savings.
    """
    try:
        from PIL import Image, ImageOps
    except ImportError:
        logger.debug("Pillow not available, skipping compression for %s", source_path)
        return None, None

    ext = source_path.rsplit(".", 1)[-1].lower() if "." in source_path else ""
    if ext not in _IMAGE_SAFE_EXT and ext:
        logger.debug("Skipping compression for unsupported image ext: %s", ext)
        return None, None

    # Skip SVG — vector, not raster
    if ext in ("svg", "svgz"):
        return None, None

    quality = int(img_cfg.get("quality", 85))
    quality = max(1, min(100, quality))  # clamp to valid JPEG range
    max_dim = int(img_cfg.get("max_dim", 1920))
    convert_to_jpeg = img_cfg.get("convert_to_jpeg", True)

    # Open and auto-orient
    try:
        img = Image.open(source_path)
        img = ImageOps.exif_transpose(img) or img
    except Exception:
        logger.warning("Failed to open image for compression: %s", source_path)
        return None, None

    # Resize if larger than max_dim on longest side
    orig_w, orig_h = img.size
    if orig_w > max_dim or orig_h > max_dim:
        ratio = min(max_dim / orig_w, max_dim / orig_h)
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Save as JPEG (use tempfile for thread safety)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")

    fd, compressed_path = tempfile.mkstemp(suffix=".jpg", dir=os.path.dirname(source_path))
    os.close(fd)
    try:
        img.save(compressed_path, "JPEG", quality=quality, optimize=True)

        # Only use compressed version if it's actually smaller
        orig_size = os.path.getsize(source_path)
        comp_size = os.path.getsize(compressed_path)
        if comp_size < orig_size:
            logger.info("Image compressed: %s (%d → %d bytes, %.0f%%)",
                        source_path, orig_size, comp_size,
                        (1 - comp_size / orig_size) * 100)
            return compressed_path, "image/jpeg"
        else:
            os.unlink(compressed_path)
            logger.debug("Compression not beneficial for %s (%d→%d), keeping original",
                         source_path, orig_size, comp_size)
            return None, None
    except Exception:
        # Clean up partial file
        try:
            os.unlink(compressed_path)
        except OSError:
            pass
        logger.warning("Image compression save failed for %s", source_path)
        return None, None


def _compress_video(source_path: str, output_dir: str,
                    video_cfg: dict,
                    timeout: int = 600) -> tuple[Optional[str], Optional[str]]:
    """Compress a video using ffmpeg.

    Transcodes to H.264/AAC in MP4 container.
    Returns (compressed_path, "video/mp4") or (None, None).
    """
    if not _ffmpeg_available():
        logger.debug("ffmpeg not available, skipping video compression for %s", source_path)
        return None, None

    bitrate = video_cfg.get("bitrate", "1M")
    max_fps = int(video_cfg.get("max_fps", 30))
    audio_bitrate = video_cfg.get("audio_bitrate", "128k")

    fd, compressed_path = tempfile.mkstemp(suffix=".mp4", dir=output_dir)
    os.close(fd)

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", source_path,
            "-c:v", "libx264",
            "-b:v", bitrate,
            "-vf", f"fps={max_fps},scale='if(gt(iw,ih),min({video_cfg.get('max_dim', 1280)},iw),-1):if(gt(ih,iw),min({video_cfg.get('max_dim', 1280)},ih),-1)',setpts=PTS",
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-movflags", "+faststart",
            compressed_path,
        ]
        subprocess.run(cmd, capture_output=True, timeout=timeout, check=True)

        orig_size = os.path.getsize(source_path)
        comp_size = os.path.getsize(compressed_path)
        if comp_size > 0 and comp_size < orig_size:
            logger.info("Video compressed: %s (%d → %d bytes, %.0f%%)",
                        source_path, orig_size, comp_size,
                        (1 - comp_size / orig_size) * 100)
            return compressed_path, "video/mp4"
        else:
            os.unlink(compressed_path)
            logger.debug("Video compression not beneficial for %s (%d→%d)",
                         source_path, orig_size, comp_size)
            return None, None
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as e:
        try:
            os.unlink(compressed_path)
        except OSError:
            pass
        logger.debug("Video compression failed for %s: %s", source_path, e)
        return None, None


def _compress_audio(source_path: str, output_dir: str,
                    audio_cfg: dict,
                    timeout: int = 600) -> tuple[Optional[str], Optional[str]]:
    """Compress an audio file using ffmpeg.

    Transcodes to MP3 (libmp3lame).
    Returns (compressed_path, "audio/mpeg") or (None, None).
    """
    if not _ffmpeg_available():
        logger.debug("ffmpeg not available, skipping audio compression for %s", source_path)
        return None, None

    bitrate = audio_cfg.get("bitrate", "128k")
    sample_rate = int(audio_cfg.get("sample_rate", 44100))

    fd, compressed_path = tempfile.mkstemp(suffix=".mp3", dir=output_dir)
    os.close(fd)

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", source_path,
            "-c:a", "libmp3lame",
            "-b:a", bitrate,
            "-ar", str(sample_rate),
            "-map_metadata", "-1",  # strip metadata to avoid bloat
            compressed_path,
        ]
        subprocess.run(cmd, capture_output=True, timeout=timeout, check=True)

        orig_size = os.path.getsize(source_path)
        comp_size = os.path.getsize(compressed_path)
        if comp_size > 0 and comp_size < orig_size:
            logger.info("Audio compressed: %s (%d → %d bytes, %.0f%%)",
                        source_path, orig_size, comp_size,
                        (1 - comp_size / orig_size) * 100)
            return compressed_path, "audio/mpeg"
        else:
            os.unlink(compressed_path)
            logger.debug("Audio compression not beneficial for %s (%d→%d)",
                         source_path, orig_size, comp_size)
            return None, None
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as e:
        try:
            os.unlink(compressed_path)
        except OSError:
            pass
        logger.debug("Audio compression failed for %s: %s", source_path, e)
        return None, None


def _ffmpeg_available() -> bool:
    """Check if ffmpeg is available on PATH (cached after first check)."""
    global _FFMPEG_AVAILABLE
    if _FFMPEG_AVAILABLE is None:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
            _FFMPEG_AVAILABLE = True
        except Exception:
            _FFMPEG_AVAILABLE = False
    return _FFMPEG_AVAILABLE
