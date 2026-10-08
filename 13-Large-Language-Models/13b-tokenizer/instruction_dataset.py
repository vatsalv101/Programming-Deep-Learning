from pathlib import Path

from torch.utils.data import Dataset


DATASET_FILENAME = "code_instructions_120k.arrow"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / DATASET_FILENAME
FORMAT_NAME = b"instruction-dataset"
FORMAT_VERSION = 3
STORED_COLUMNS = ("instruction", "input", "output")


def dataset_not_found_message(path):
    return (
        "Dataset file missing. Put "
        + DATASET_FILENAME
        + " next to this notebook and run the cell again."
    )


def find_dataset_path(path=None):
    search_paths = []

    if path is not None:
        search_paths.append(Path(path).expanduser())

    search_paths.append(DEFAULT_OUTPUT_PATH)

    repo_root = Path(__file__).resolve().parent.parent
    search_paths.extend(
        [
            repo_root / "13a-dataset-exploration" / DATASET_FILENAME,
            repo_root / "13c-llm" / DATASET_FILENAME,
        ]
    )

    for candidate in search_paths:
        if candidate.exists():
            return candidate
    return search_paths[0]


class InstructionDataset(Dataset):
    """Memory-mapped PyTorch dataset for instruction-tuning text."""

    def __init__(self, path=None):
        self.path = find_dataset_path(path)
        self._open()

    def _open(self):
        last_error = None

        for candidate in [self.path, *self._candidate_paths()]:
            candidate = Path(candidate).expanduser()
            if not candidate.exists():
                last_error = FileNotFoundError(dataset_not_found_message(candidate))
                continue

            try:
                import pyarrow as pa

                self._source = pa.memory_map(str(candidate), "r")
                self._table = pa.ipc.open_file(self._source).read_all()
                metadata = self._table.schema.metadata or {}
                if metadata.get(b"format") != FORMAT_NAME:
                    raise ValueError(str(candidate) + " is not an instruction dataset.")
                if metadata.get(b"version") != str(FORMAT_VERSION).encode():
                    raise ValueError("Unsupported dataset version.")

                self.path = candidate
                self.column_names = tuple(self._table.column_names)
                return
            except (FileNotFoundError, OSError, ValueError) as exc:
                last_error = exc
                if hasattr(self, "_source"):
                    try:
                        self._source.close()
                    except Exception:
                        pass
                if hasattr(self, "_table"):
                    del self._table

        raise last_error or FileNotFoundError(dataset_not_found_message(self.path))

    def _candidate_paths(self):
        repo_root = Path(__file__).resolve().parent.parent
        return [
            DEFAULT_OUTPUT_PATH,
            repo_root / "13a-dataset-exploration" / DATASET_FILENAME,
            repo_root / "13c-llm" / DATASET_FILENAME,
        ]

    def __len__(self):
        return self._table.num_rows

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        row = {}
        for name in self.column_names:
            row[name] = self._table[name][index].as_py()
        return row

    def __getstate__(self):
        return {"path": self.path}

    def __setstate__(self, state):
        self.path = state["path"]
        self._open()

    def close(self):
        """Release the memory map, allowing the Arrow file to be replaced."""
        if hasattr(self, "_table"):
            del self._table
        if hasattr(self, "_source"):
            self._source.close()
            del self._source

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
