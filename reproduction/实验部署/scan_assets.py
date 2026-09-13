"""Read every inventoried file; distinguish semantic parsing from binary inspection."""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import ast
import csv
import hashlib
import io
import json
import os
import re
import struct
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
TEXT = {'.md', '.txt', '.tex', '.py', '.sh', '.bat', '.json', '.jsonl', '.csv',
        '.tsv', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.log', '.bib', '.rst',
        '.cls', '.xml', '.html', '.ipynb', '.out', '.h', '.pyx', '.pxd', '.c'}
PDF_LOCK = threading.Lock()


def scope(path):
    if '/.git/' in path or '/vendor/' in path or path.startswith('基线模型/参考实现/'):
        return 'third_party_or_git'
    if 'snapshot' in path or '/captured_code/' in path or '/source_code/' in path:
        return 'frozen_code_snapshot'
    if path.startswith('归档_相关材料/') or path.startswith('论文修订版_'):
        return 'archive_or_previous_manuscript'
    if path.startswith('数据集/'):
        return 'raw_or_provider_data'
    return 'project'


def inspect_json(obj):
    result = {}
    if isinstance(obj, dict):
        result['keys'] = list(obj)[:60]
        selected = {}
        for k, v in obj.items():
            if re.search(r'status|passed|complete|fail|reject|accept|selected|decision|reason|note|limit|not_done|next|finding|scope|total|count|rows|cells|folds|metric|version|method|model|support|budget', k, re.I):
                if isinstance(v, (str, int, float, bool)) or v is None:
                    selected[k] = v[:600] if isinstance(v, str) else v
                elif isinstance(v, list):
                    selected[k] = {'length': len(v), 'sample': str(v[:2])[:450]}
                elif isinstance(v, dict):
                    selected[k] = {'length': len(v), 'keys': list(v)[:15]}
        result['selected_fields'] = selected
        result['container_lengths'] = {k: len(v) for k, v in obj.items() if isinstance(v, (dict, list))}
    elif isinstance(obj, list):
        result['items'] = len(obj)
        result['first_type'] = type(obj[0]).__name__ if obj else None
    else:
        result['scalar'] = obj
    return result


def scan(row):
    rel = row['path']
    p = ROOT / rel
    ext = p.suffix.lower()
    r = {'path': rel, 'size_bytes': int(row['size_bytes']), 'scope': scope(rel),
         'suffix': ext, 'review_mode': 'binary_header_and_full_file_hash'}
    try:
        digest = hashlib.sha256()
        blocks = [] if ext in TEXT else None
        head = b''
        with p.open('rb') as f:
            while True:
                block = f.read(4 * 1024 * 1024)
                if not block:
                    break
                if not head:
                    head = block[:256]
                digest.update(block)
                if blocks is not None:
                    blocks.append(block)
        r['sha256'] = digest.hexdigest()
        if blocks is not None:
            raw = b''.join(blocks)
            try:
                text = raw.decode('utf-8-sig')
                r['encoding'] = 'utf-8'
            except UnicodeDecodeError:
                try:
                    text = raw.decode('gb18030')
                    r['encoding'] = 'gb18030'
                except UnicodeDecodeError:
                    text = raw.decode('utf-8', errors='replace')
                    r['encoding'] = 'utf-8-with-replacements'
            r['lines'] = text.count('\n') + 1
            r['review_mode'] = 'full_text_read_and_indexed'
            if ext == '.json':
                r.update(inspect_json(json.loads(text)))
                r['review_mode'] = 'full_json_parsed'
            elif ext in {'.csv', '.tsv'}:
                reader = csv.reader(io.StringIO(text), delimiter='\t' if ext == '.tsv' else ',')
                cols = next(reader, [])
                count = 0
                widths = Counter()
                for values in reader:
                    count += 1
                    widths[len(values)] += 1
                r.update(columns=cols, rows=count, row_width_counts=dict(widths))
                r['review_mode'] = 'full_table_rows_parsed'
            elif ext == '.jsonl':
                count = 0
                errors = 0
                for line in text.splitlines():
                    if line.strip():
                        count += 1
                        try:
                            json.loads(line)
                        except (ValueError, TypeError):
                            errors += 1
                r.update(records=count, malformed_records=errors)
                r['review_mode'] = 'full_jsonl_records_parsed'
            elif ext == '.py':
                tree = ast.parse(text)
                r['definitions'] = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
                r['docstring'] = (ast.get_docstring(tree) or '')[:1600]
                r['review_mode'] = 'full_python_source_parsed_ast'
            elif ext in {'.md', '.tex', '.txt', '.rst'}:
                r['headings'] = [x for x in text.splitlines() if x.startswith('#') or re.match(r'\\(?:sub)*section', x)][:100]
            if ext == '.log':
                lines = text.splitlines()
                r['last_lines'] = lines[-4:]
                r['error_lines'] = [x[:250] for x in lines if re.search(r'Traceback|ERROR|RuntimeError|FAILED|Error:', x)][:8]
        elif ext == '.npz':
            arrays = []
            with zipfile.ZipFile(p) as z:
                for name in z.namelist():
                    if name.endswith('.npy'):
                        with z.open(name) as f:
                            magic = f.read(8)
                            if magic[:6] != b'\x93NUMPY':
                                arrays.append({'name': name, 'error': 'not_npy'})
                                continue
                            width = 2 if magic[6] == 1 else 4
                            length = int.from_bytes(f.read(width), 'little')
                            if length > 1024 * 1024:
                                raise ValueError('unexpectedly long npy header')
                            desc = ast.literal_eval(f.read(length).decode('latin1').strip())
                            arrays.append({'name': name, 'shape': desc.get('shape'), 'dtype': desc.get('descr')})
                    else:
                        arrays.append({'name': name})
            r['arrays'] = arrays
            r['review_mode'] = 'array_headers_and_full_file_hash_not_all_values'
        elif ext in {'.zip', '.pt', '.pth', '.xlsx'} and zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:
                entries = z.infolist()
                r['archive_members'] = len(entries)
                r['archive_uncompressed_bytes'] = sum(x.file_size for x in entries)
                if ext == '.zip':
                    r['archive_manifest'] = [{'name': x.filename, 'bytes': x.file_size} for x in entries]
                else:
                    r['archive_names_sample'] = [x.filename for x in entries[:25]]
            r['review_mode'] = 'archive_directory_and_full_file_hash_not_all_members'
        elif ext == '.pdf':
            import pymupdf
            with PDF_LOCK, pymupdf.open(p) as doc:
                text = '\n'.join(page.get_text() for page in doc)
                r['pdf_pages'] = len(doc)
                r['pdf_text_chars'] = len(text)
                target = OUT / 'pdf_text' / (r['sha256'] + '.txt')
                target.parent.mkdir(exist_ok=True)
                target.write_text(text)
                r['pdf_text_file'] = str(target.relative_to(OUT))
            r['review_mode'] = 'all_pdf_pages_text_extracted'
        elif ext in {'.png', '.jpg', '.jpeg'}:
            from PIL import Image
            with Image.open(p) as im:
                r['image_size'] = im.size
                r['image_mode'] = im.mode
            r['review_mode'] = 'image_header_and_full_file_hash'
        else:
            r['file_magic_hex'] = head[:24].hex()
        r['read_ok'] = True
    except Exception as e:
        r['read_ok'] = False
        r['error'] = type(e).__name__ + ': ' + str(e)[:300]
    return r


def main():
    start = time.time()
    with (OUT / '全量文件索引.csv').open() as f:
        inventory = list(csv.DictReader(f))
    done = 0
    totals = Counter()
    modes = Counter()
    failed = []
    sizes = 0
    with (OUT / '逐文件程序审查.jsonl').open('w') as output:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for r in pool.map(scan, inventory, chunksize=16):
                output.write(json.dumps(r, ensure_ascii=False) + '\n')
                done += 1
                sizes += r['size_bytes']
                totals[r['suffix']] += 1
                modes[r['review_mode']] += 1
                if not r['read_ok']:
                    failed.append({'path': r['path'], 'error': r['error']})
                if done % 2000 == 0:
                    output.flush()
                    progress = {'done': done, 'total': len(inventory), 'bytes_read': sizes, 'elapsed_s': round(time.time() - start, 1), 'errors': len(failed)}
                    (OUT / '扫描进度.json').write_text(json.dumps(progress, ensure_ascii=False))
                    print(json.dumps(progress), flush=True)
    summary = {'done': done, 'total': len(inventory), 'bytes_read': sizes, 'elapsed_s': time.time() - start,
               'review_modes': dict(modes), 'failures': failed}
    (OUT / '程序审查摘要.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (OUT / '扫描进度.json').write_text(json.dumps({'status': 'complete', **summary}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
