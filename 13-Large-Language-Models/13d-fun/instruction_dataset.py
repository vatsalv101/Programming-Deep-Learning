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


class InstructionDataset(Dataset):
    """Memory-mapped PyTorch dataset for instruction-tuning text."""

    def __init__(self, path=DEFAULT_OUTPUT_PATH):
        self.path = Path(path)
        self._open()

    def _open(self):
        if not self.path.exists():
            raise FileNotFoundError(dataset_not_found_message(self.path))

        import pyarrow as pa

        self._source = pa.memory_map(str(self.path), "r")
        self._table = pa.ipc.open_file(self._source).read_all()

        metadata = self._table.schema.metadata or {}
        if metadata.get(b"format") != FORMAT_NAME:
            raise ValueError(str(self.path) + " is not an instruction dataset.")
        if metadata.get(b"version") != str(FORMAT_VERSION).encode():
            raise ValueError("Unsupported dataset version.")

        self.column_names = tuple(self._table.column_names)

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
