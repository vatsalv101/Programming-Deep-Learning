from pathlib import Path
from urllib.request import Request, urlopen
import shutil
import zipfile

import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from tqdm.auto import tqdm


MEAN = (0.5,0.5,0.5)
STD = (0.5,0.5,0.5)

AFHQ_URL = "https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=1"
AFHQ_DIR_NAME = "afhq_v2"

data_path = Path.cwd().resolve().parent.parent / "data"
data_path.mkdir(parents=True, exist_ok=True)
data_root = str(data_path)


def download_file(url, out_path, chunk_size=1024 * 1024):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urlopen(req) as response:
        total = response.headers.get("Content-Length")
        total = int(total) if total is not None else None

        with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading AFHQ v2") as pbar:
            while chunk := response.read(chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))


def valid_afhq_dir(path):
    path = Path(path)
    return (path / "train" / "cat").exists() and ((path / "val" / "cat").exists() or (path / "test" / "cat").exists())


def detect_zip_prefix(zip_ref):
    names = [
        info.filename.replace("\\", "/")
        for info in zip_ref.infolist()
        if not info.is_dir() and not info.filename.startswith("__MACOSX")
    ]

    if any(name.startswith("train/") for name in names):
        return ""

    top_levels = {name.split("/", 1)[0] for name in names if "/" in name}

    for top in top_levels:
        prefix = top + "/"
        has_train = any(name.startswith(prefix + "train/") for name in names)
        has_val = any(name.startswith(prefix + "val/") for name in names)
        has_test = any(name.startswith(prefix + "test/") for name in names)

        if has_train and (has_val or has_test):
            return prefix

    raise RuntimeError("Could not detect AFHQ folder structure inside zip.")


def extract_zip(zip_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_resolved = out_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        prefix = detect_zip_prefix(zip_ref)

        for member in tqdm(zip_ref.infolist(), desc="Extracting AFHQ v2", unit="file"):
            filename = member.filename.replace("\\", "/")

            if filename.startswith("__MACOSX"):
                continue

            if prefix:
                if not filename.startswith(prefix):
                    continue
                filename = filename[len(prefix):]

            if not filename:
                continue

            target = (out_dir / filename).resolve()

            try:
                target.relative_to(out_dir_resolved)
            except ValueError:
                raise RuntimeError(f"Unsafe zip path: {member.filename}")

            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zip_ref.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def download_and_extract_afhq_v2(root):
    root = Path(root)
    dataset_dir = root / AFHQ_DIR_NAME
    zip_path = root / "afhq_v2.zip"

    if valid_afhq_dir(dataset_dir):
        return dataset_dir

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    if not zip_path.exists():
        download_file(AFHQ_URL, zip_path)
    else:
        print(f"Using existing zip file: {zip_path}")

    try:
        extract_zip(zip_path, dataset_dir)
    except zipfile.BadZipFile:
        zip_path.unlink(missing_ok=True)
        download_file(AFHQ_URL, zip_path)
        extract_zip(zip_path, dataset_dir)

    zip_path.unlink(missing_ok=True)

    if not valid_afhq_dir(dataset_dir):
        raise RuntimeError(
            f"AFHQ extraction failed. Expected:\n"
            f"{dataset_dir / 'train' / 'cat'}\n"
            f"{dataset_dir / 'val' / 'cat'} or {dataset_dir / 'test' / 'cat'}"
        )

    return dataset_dir


def image_files(root):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in exts)


def resize_afhq_dataset(source_dir, root, image_size):
    source_dir = Path(source_dir)
    cache_dir = Path(root) / f"{AFHQ_DIR_NAME}_{image_size}"
    done_file = cache_dir / ".complete"

    if done_file.exists() and valid_afhq_dir(cache_dir):
        return cache_dir

    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    files = image_files(source_dir)

    if len(files) == 0:
        raise RuntimeError(f"No images found in {source_dir}")

    for src in tqdm(files, desc=f"Creating resized AFHQ cache {image_size}x{image_size}"):
        rel = src.relative_to(source_dir)
        dst = cache_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(src) as img:
            #img = img.convert("L")  # grayscale
            img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
            img.save(dst)

    done_file.write_text("ok")

    if not valid_afhq_dir(cache_dir):
        raise RuntimeError(
            f"Resized AFHQ cache invalid. Expected:\n"
            f"{cache_dir / 'train' / 'cat'}\n"
            f"{cache_dir / 'val' / 'cat'} or {cache_dir / 'test' / 'cat'}"
        )

    return cache_dir


class AFHQv2(Dataset):
    def __init__(self, root, class_name="cat"):
        root = Path(root)
        class_dir = root / class_name

        if not class_dir.exists():
            raise RuntimeError(f"Class directory not found: {class_dir}")

        self.paths = image_files(class_dir)
        self.class_to_idx = {class_name: 0}

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {class_dir}")

        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(MEAN, STD),
        ])

        xs = []

        for path in self.paths:
            with Image.open(path) as img:
                #img = img.convert("L")  # grayscale
                xs.append(transform(img))

        self.x = torch.stack(xs)
        self.y = torch.zeros(len(xs), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_afhq_loaders(batch_size, num_workers=0, image_size=32, class_name="cat", seed=0):
    raw_dir = download_and_extract_afhq_v2(data_root)
    dataset_dir = resize_afhq_dataset(raw_dir, data_root, image_size)

    eval_split = "val" if (dataset_dir / "val" / class_name).exists() else "test"

    train_data = AFHQv2(dataset_dir / "train", class_name=class_name)
    eval_data = AFHQv2(dataset_dir / eval_split, class_name=class_name)

    val_size = len(eval_data) // 2
    test_size = len(eval_data) - val_size

    val_data, test_data = random_split(eval_data, [val_size, test_size], generator=torch.Generator().manual_seed(seed))

    kwargs = {"num_workers": num_workers}

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

    print(f"AFHQ raw dir: {raw_dir}")
    print(f"AFHQ resized dir: {dataset_dir}")
    print(f"AFHQ image size: {image_size}x{image_size}")
    print(f"AFHQ {class_name} train: {len(train_data)}")
    print(f"AFHQ {class_name} val: {len(val_data)}")
    print(f"AFHQ {class_name} test: {len(test_data)}")

    return train_loader, val_loader, test_loader