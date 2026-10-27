import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from fastapi import HTTPException, UploadFile, status

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_VIDEO_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024   # 20 MB


def to_media_url(file_path: Optional[str]) -> Optional[str]:
    """
    Transforms any internal filesystem path into a safe, client-facing static URL.
    Never exposes arbitrary local filesystem paths to the client.
    """
    if not file_path:
        return None

    # Normalize Windows backslashes
    norm_path = str(file_path).replace("\\", "/")

    # Check for violations subdirectory
    if "data/output/violations/" in norm_path:
        filename = norm_path.split("data/output/violations/")[-1]
        return f"/media/violations/{filename}"

    # Check for general output directory
    if "data/output/" in norm_path:
        filename = norm_path.split("data/output/")[-1]
        return f"/media/output/{filename}"

    # If it already looks like a media path
    if norm_path.startswith("/media/"):
        return norm_path

    # Fallback to basename
    base_name = Path(norm_path).name
    if "violations" in norm_path.lower():
        return f"/media/violations/{base_name}"
    return f"/media/output/{base_name}"


def generate_safe_filename(prefix: str, extension: str) -> str:
    """
    Generates a unique, collision-resistant filename using timestamp and UUID slice.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand_suffix = uuid.uuid4().hex[:6]
    clean_ext = extension.lstrip(".")
    return f"{prefix}_{ts}_{rand_suffix}.{clean_ext}"


def validate_file_extension(filename: Optional[str], allowed_extensions: set) -> str:
    """
    Validates that the file has one of the allowed extensions.
    Returns the lowercased extension.
    """
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename."
        )

    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        allowed_list = ", ".join(sorted(allowed_extensions))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {allowed_list}"
        )
    return ext


async def save_upload_stream(
    upload_file: UploadFile,
    dest_path: str,
    max_size_bytes: int = MAX_VIDEO_SIZE_BYTES
) -> int:
    """
    Streams an uploaded file to disk in chunks to avoid high RAM consumption.
    Enforces maximum size limit and aborts if exceeded.
    """
    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    bytes_written = 0
    chunk_size = 1024 * 1024  # 1MB chunks

    try:
        with open(dest_path, "wb") as f_out:
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_size_bytes:
                    # Clean up partial file
                    f_out.close()
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    max_mb = max_size_bytes / (1024 * 1024)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {max_mb:.0f} MB."
                    )
                f_out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(exc)}"
        )
    finally:
        await upload_file.seek(0)

    return bytes_written
