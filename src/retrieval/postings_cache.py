"""Disposable, bounded SSD cache for exact transformed postings.

Keys include immutable parent identity, statistics, switches and implementation.
NPZ checksums detect damaged entries; interrupted writes are never published.
Eviction only removes cache files, never representations or query checkpoints.
"""
from __future__ import annotations

import fcntl
import os
import shutil
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from scipy import sparse

from src.artifacts import file_hash, read_json
from src.config import digest
from src.retrieval.index import InvertedIndex


class PostingsCache:
    def __init__(self, directory, identity, stats, switches, max_bytes, min_free_bytes=32 * 2**30):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stats, self.switches = stats, switches
        self.max_bytes, self.min_free_bytes = max_bytes, min_free_bytes
        code = Path(__file__).resolve()
        implementation = {p.name: file_hash(p) for p in
                          (code, code.with_name('index.py'), code.parents[1] / 'scoring.py')}
        self.prefix = digest({'identity': identity, 'statistics': stats.to_dict(),
                              'switches': asdict(switches), 'implementation': implementation})
        self.db_path = self.directory / 'entries.sqlite'
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, bytes INTEGER, used REAL)')
            db.execute('CREATE INDEX IF NOT EXISTS eviction_order ON entries(used)')
        self.hits = self.misses = 0

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.db_path, timeout=120)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def load(path):
        with np.load(path, allow_pickle=False) as data:
            index = InvertedIndex.__new__(InvertedIndex)
            index.doc_ids = data['ids']
            index.postings = sparse.csc_matrix((data['data'], data['indices'], data['indptr']),
                                               shape=tuple(data['shape']))
            index.postings.check_format(full_check=True)
            if len(index.doc_ids) != index.postings.shape[0]:
                raise ValueError('Cached document IDs do not match postings')
            return index

    def get(self, shard):
        key = digest({'prefix': self.prefix, 'shard': shard})
        path = self.directory / (key + '.npz')
        # Builders for a shared corpus/configuration cooperate across evaluations.
        with (self.directory / (key + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                index = self.load(path)
                self.hits += 1
                # Touch at most once per minute to limit SQLite write traffic.
                with self.connect() as db:
                    now = time.time()
                    db.execute('UPDATE entries SET used=? WHERE key=? AND used<?', (now, key, now - 60))
                return index
            except (OSError, ValueError, KeyError, BadZipFile, EOFError):
                pass
            self.misses += 1
            index = InvertedIndex(sparse.load_npz(shard['vectors']), read_json(shard['ids']),
                                  self.stats, self.switches)
            estimated = sum(a.nbytes for a in (index.doc_ids, index.postings.data,
                                               index.postings.indices, index.postings.indptr)) + 8192
            if estimated > self.max_bytes or shutil.disk_usage(self.directory).free < self.min_free_bytes + estimated:
                return index
            fd, temporary = tempfile.mkstemp(prefix='.building-', suffix='.npz', dir=self.directory)
            try:
                with os.fdopen(fd, 'wb') as handle:
                    np.savez(handle, ids=index.doc_ids, data=index.postings.data,
                             indices=index.postings.indices, indptr=index.postings.indptr,
                             shape=index.postings.shape)
                size = os.path.getsize(temporary)
                with self.connect() as db:
                    db.execute('BEGIN IMMEDIATE')
                    db.execute('DELETE FROM entries WHERE key=?', (key,))
                    total = db.execute('SELECT COALESCE(SUM(bytes),0) FROM entries').fetchone()[0]
                    while total + size > self.max_bytes:
                        victim = db.execute('SELECT key,bytes FROM entries ORDER BY used LIMIT 1').fetchone()
                        if victim is None:
                            break
                        (self.directory / (victim[0] + '.npz')).unlink(missing_ok=True)
                        db.execute('DELETE FROM entries WHERE key=?', (victim[0],))
                        total -= victim[1]
                    os.replace(temporary, path)
                    db.execute('INSERT INTO entries VALUES (?,?,?)', (key, size, time.time()))
                return index
            finally:
                Path(temporary).unlink(missing_ok=True)
