# ejvpaths.py

import shutil
from pathlib import Path

from tqdm import tqdm


def find_paths(
    directory: str | Path,
    filetypes: str | list[str],
) -> list[Path]:
    """
    Find files of given type(s) at directory
    Args:
        directory : path to search, relative to project root
            ex. data/raws
        filetypes : extension string or list of them to search for
            ex. [".png", ".jpg"] leading dot is optional
    Retturn:
        sorted list of matching paths
    """
    searchdir = Path(directory)
    if not searchdir.is_dir():
        raise NotADirectoryError(f"Not a directory: {searchdir.resolve()}")

    # normalizes filetypes to list of strings
    if isinstance(filetypes, str):
        filetypes = [filetypes]
    # normalizes extensions to have leading dot and be lowercase
    exts = {f".{ft.lstrip('.')}".lower() for ft in filetypes}

    # search and filepaths
    matches = sorted(
        p for p in searchdir.iterdir() if p.is_file() and p.suffix.lower() in exts
    )

    return matches


def move_files(
    filepaths: Path | list[Path],
    destination: str | Path,
    show=False,
) -> None:
    """
    move one or more files to destination directory
    args:
        filepaths : single path or list of paths to move
        destination : target directory, created if not exist
    """
    # normalizes filepaths to list of Paths
    if isinstance(filepaths, Path):
        filepaths = [filepaths]

    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)

    for fp in tqdm(filepaths, desk="Moving files", unit="file", dynamic_ncols=True):
        shutil.move(str(fp), dest / fp.name)
    return
