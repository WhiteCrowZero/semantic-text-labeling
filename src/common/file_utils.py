import hashlib
from typing import List

import pandas as pd
from fastapi import UploadFile


def parse_text_file(upload: UploadFile, column_name: str = "text") -> List[str]:
    filename = upload.filename.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(upload.file)
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(upload.file)
    else:
        raise ValueError("不支持的文件格式，只支持 CSV/Excel")

    if column_name not in df.columns:
        raise ValueError(f"列名 '{column_name}' 不存在，请检查文件内容")

    texts = df[column_name].astype(str).tolist()
    return texts

def generate_hash(texts: List[str]) -> str:
    joined = "\n".join(texts)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()
