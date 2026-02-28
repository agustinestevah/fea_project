import pandas as pd
import numpy as np
import torch
import os
import pickle
import shutil
from sentence_transformers import SentenceTransformer, util, InputExample, losses, CrossEncoder
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import List, Tuple, Any, Mapping, Iterable, Dict, Literal, Union, Optional
import plotly.graph_objects as go
import gc


# ================================================================
# SEEN-INDEX HELPERS  (used by generate_valid_pairs for cross-round
# tracking of already-consumed pair indices)
# ================================================================

def _load_seen(path: str) -> np.ndarray:
    """Load a sorted int64 numpy array from *path*, or return empty."""
    if os.path.exists(path):
        arr = np.load(path)
        return np.sort(arr.astype(np.int64))
    return np.array([], dtype=np.int64)


def _save_seen(path: str, indices: np.ndarray) -> None:
    """Save a sorted int64 numpy array to *path*."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.sort(indices.astype(np.int64)))


def _sample_excluding(
    rng: np.random.Generator,
    universe_size: int,
    excluded_sorted: np.ndarray,
    sample_size: int,
) -> np.ndarray:
    """
    Sample *sample_size* unique indices from ``[0, universe_size)``
    **excluding** the indices in *excluded_sorted* (must be sorted).

    Uses an iterative search-sorted offset correction that converges
    in O(log(M/N)) steps where M = len(excluded).
    """
    n_available = universe_size - len(excluded_sorted)
    if n_available <= 0:
        return np.array([], dtype=np.int64)
    if sample_size > n_available:
        sample_size = n_available

    # Sample from the "compressed" space [0, n_available)
    compressed = np.sort(rng.choice(n_available, size=sample_size, replace=False))

    # Map to original space via iterative offset correction
    expanded = compressed.astype(np.int64)
    for _ in range(50):  # usually converges in 2-3 iterations
        offsets = np.searchsorted(excluded_sorted, expanded, side='right')
        new_expanded = (compressed + offsets).astype(np.int64)
        if np.array_equal(new_expanded, expanded):
            break
        expanded = new_expanded

    return expanded


# Sklearn imports for the regression/prediction steps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, precision_recall_curve
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# SETUP

def generate_valid_pairs(
    df_p: pd.DataFrame,
    df_sc: pd.DataFrame,
    id_col: str = 'sentence_id',
    text_col: str = 'sentence',
    max_pairs: int = 500000,
    random_seed: int = 42,
    embedding_cache: Optional[Dict[str, np.ndarray]] = None,
    threshold_bb: Optional[float] = None,
    threshold_bs: Optional[float] = None,
    batch_size: int = 512,
    sample_n_bb: Optional[int] = None,
    sample_n_bs: Optional[int] = None,
    top_k_bs: Optional[int] = None,
    exclude_labeled_csv: Optional[str] = "labeled_pairs/llm_labeled_pairs.csv",
    seen_indices_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate valid pairwise combinations from premise and conclusion dataframes.

    Has two modes of operation:

    1. **Random sampling** (default, original behaviour):
       When ``embedding_cache`` / thresholds are *not* supplied, pairs are
       randomly sampled up to ``max_pairs``.

    2. **Threshold-filtered** (memory-efficient):
       When ``embedding_cache``, ``threshold_bb`` **and** ``threshold_bs`` are
       all supplied, the function scans *all* candidate pairs in batches using
       matrix dot-products and keeps only those whose cosine similarity exceeds
       the respective threshold.  The full similarity matrix is **never
       materialised**; only one batch (``batch_size × n``) slice is held in
       memory at a time.

    Rules (both modes):
    - Premise-Premise pairs (within df_p): allowed if not both Speech
    - Conclusion-Conclusion pairs (within df_sc): allowed if not both Speech
    - Premise-Conclusion pairs: NOT allowed (no mixing)
    - Speech-Speech pairs: NOT allowed (filtered out)
    - Book-Book pairs: allowed
    - Book-Speech pairs: allowed
    
    ID format expected: [B/S]XXXXX[p/sc]
    - First character: B (Book) or S (Speech)
    - Last 1-2 characters: p (premise) or sc (conclusion)
    
    Parameters
    ----------
    df_p : pd.DataFrame
        Dataframe containing premise sentences with an ID column.
    df_sc : pd.DataFrame
        Dataframe containing conclusion sentences with an ID column.
    id_col : str, default='sentence_id'
        Name of the column containing sentence IDs.
    text_col : str, default='sentence'
        Name of the column containing the text/sentence content.
    max_pairs : int, default=500000
        Maximum number of pairs to sample (ignored when thresholds are given).
    random_seed : int, default=42
        Random seed for reproducibility.
    embedding_cache : dict, optional
        Pre-computed embeddings ``{id_str: np.ndarray}``.  Required for
        threshold-filtered mode.
    threshold_bb : float, optional
        Minimum cosine similarity to keep a Book-Book pair.
    threshold_bs : float, optional
        Minimum cosine similarity to keep a Book-Speech pair.
    batch_size : int, default=512
        Number of rows per batch when computing cosine similarities.
    sample_n_bb : int, optional
        If set, randomly sample this many Book-Book pairs from the final
        combined results.  Applied after both groups are processed.
    sample_n_bs : int, optional
        If set, randomly sample this many Book-Speech pairs from the
        top ``top_k_bs`` BS pairs (by cosine similarity).
    top_k_bs : int, optional
        If set together with ``sample_n_bs``, keep only the top
        ``top_k_bs`` BS pairs (by cosine similarity) before sampling.
        Also used *inside* each group scan to cap intermediate BS results
        so they don't blow up memory.
    exclude_labeled_csv : str or None, default="labeled_pairs/llm_labeled_pairs.csv"
        Path to a CSV of already-labeled pairs (``sentence_id_1``,
        ``sentence_id_2``).  Any matching pairs are removed from the
        output.  Set to ``None`` to disable filtering.
    seen_indices_dir : str or None, default=None
        Directory for persisting consumed linear-index files across rounds.
        When set, each call loads previously consumed indices so the same
        pair is never generated twice, even across separate invocations.
        Four files are maintained (one per segment):
        ``{dir}/seen_premises_bb.npy``, ``seen_premises_bs.npy``,
        ``seen_conclusions_bb.npy``, ``seen_conclusions_bs.npy``.
        Set to ``None`` to disable (original behaviour).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['id1', 'id2']`` (random sampling mode)
        **or** ``['id1', 'id2', 'cosine_sim', 'pair_type']``
        (threshold-filtered mode).  Text columns are NOT included;
        look them up on demand via ``df_clause`` when needed.
    """
    np.random.seed(random_seed)
    
    def get_source_type(text_id: str) -> str:
        """Extract source type (B or S) from ID."""
        if isinstance(text_id, str) and len(text_id) > 0:
            return text_id[0].upper()
        return None

    # ================================================================
    # THRESHOLD-FILTERED PATH  (memory-efficient batch scanning)
    # ================================================================
    if embedding_cache is not None and threshold_bb is not None and threshold_bs is not None:
        import gc

        df_all = pd.concat([df_p, df_sc]).drop_duplicates(subset=id_col)
        id_to_text = dict(zip(df_all[id_col], df_all[text_col]))

        def _filtered_pairs_within_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
            ids = df[id_col].tolist()
            book_ids = [x for x in ids if get_source_type(x) == 'B']
            speech_ids = [x for x in ids if get_source_type(x) == 'S']
            n_book = len(book_ids)
            n_speech = len(speech_ids)

            book_id_arr = np.array(book_ids)
            speech_id_arr = np.array(speech_ids) if speech_ids else np.array([], dtype=object)

            # --- build normalised book embedding matrix (float16 to save RAM) ---
            book_embs = np.stack([embedding_cache[str(x)] for x in book_ids])
            book_norms = np.linalg.norm(book_embs, axis=1, keepdims=True)
            book_embs_n = (book_embs / np.maximum(book_norms, 1e-10)).astype(np.float16)
            del book_embs, book_norms
            gc.collect()

            # ---------- B-B pairs (upper triangle only) ----------
            bb_i1_parts, bb_i2_parts, bb_sim_parts = [], [], []
            total_bb = n_book * (n_book - 1) // 2
            print(f"    [{label}] Scanning {total_bb:,} B-B candidates in batches of {batch_size}...")

            for start in range(0, n_book, batch_size):
                end = min(start + batch_size, n_book)
                # Cast batch to float32 for the matmul, keep the full matrix in float16
                sims = book_embs_n[start:end].astype(np.float32) @ book_embs_n[start:].astype(np.float32).T

                # Mask diagonal + lower triangle (j_global must be > i_global)
                for i_loc in range(end - start):
                    sims[i_loc, :i_loc + 1] = -2.0

                rows, cols = np.where(sims >= threshold_bb)
                if len(rows) > 0:
                    bb_i1_parts.append(start + rows)
                    bb_i2_parts.append(start + cols)
                    bb_sim_parts.append(sims[rows, cols])
                del sims

            if bb_i1_parts:
                bb_i1 = np.concatenate(bb_i1_parts)
                bb_i2 = np.concatenate(bb_i2_parts)
                bb_sims = np.concatenate(bb_sim_parts)
                df_bb = pd.DataFrame({
                    'id1': book_id_arr[bb_i1],
                    'id2': book_id_arr[bb_i2],
                    'cosine_sim': bb_sims,
                    'pair_type': 'BB',
                })
                del bb_i1, bb_i2, bb_sims
            else:
                df_bb = pd.DataFrame(columns=['id1', 'id2', 'cosine_sim', 'pair_type'])

            del bb_i1_parts, bb_i2_parts, bb_sim_parts
            gc.collect()
            print(f"    [{label}] Found {len(df_bb):,} B-B pairs above threshold {threshold_bb:.4f}")

            # ---------- B-S pairs (double-batched: batch books AND speech) ----------
            if n_speech > 0:
                # Build normalised speech embedding matrix (float16)
                speech_embs = np.stack([embedding_cache[str(x)] for x in speech_ids])
                speech_norms = np.linalg.norm(speech_embs, axis=1, keepdims=True)
                speech_embs_n = (speech_embs / np.maximum(speech_norms, 1e-10)).astype(np.float16)
                del speech_embs, speech_norms
                gc.collect()

                bs_i1_parts, bs_i2_parts, bs_sim_parts = [], [], []
                total_bs = n_book * n_speech
                # Use a smaller speech-side chunk to keep peak memory low
                speech_chunk = max(batch_size * 4, 2048)
                print(f"    [{label}] Scanning {total_bs:,} B-S candidates  "
                      f"(book batch={batch_size}, speech chunk={speech_chunk})...")

                for b_start in range(0, n_book, batch_size):
                    b_end = min(b_start + batch_size, n_book)
                    book_batch = book_embs_n[b_start:b_end].astype(np.float32)

                    for s_start in range(0, n_speech, speech_chunk):
                        s_end = min(s_start + speech_chunk, n_speech)
                        sims = book_batch @ speech_embs_n[s_start:s_end].astype(np.float32).T

                        rows, cols = np.where(sims >= threshold_bs)
                        if len(rows) > 0:
                            bs_i1_parts.append(b_start + rows)
                            bs_i2_parts.append(s_start + cols)
                            bs_sim_parts.append(sims[rows, cols])
                        del sims

                    del book_batch

                del speech_embs_n
                gc.collect()

                if bs_i1_parts:
                    bs_i1 = np.concatenate(bs_i1_parts)
                    bs_i2 = np.concatenate(bs_i2_parts)
                    bs_sims = np.concatenate(bs_sim_parts)
                    df_bs = pd.DataFrame({
                        'id1': book_id_arr[bs_i1],
                        'id2': speech_id_arr[bs_i2],
                        'cosine_sim': bs_sims,
                        'pair_type': 'BS',
                    })
                    del bs_i1, bs_i2, bs_sims
                else:
                    df_bs = pd.DataFrame(columns=['id1', 'id2', 'cosine_sim', 'pair_type'])

                del bs_i1_parts, bs_i2_parts, bs_sim_parts
                gc.collect()
                print(f"    [{label}] Found {len(df_bs):,} B-S pairs above threshold {threshold_bs:.4f}")

                # --- Trim BS to top_k_bs right away to save memory ---
                if top_k_bs is not None and len(df_bs) > top_k_bs:
                    df_bs = df_bs.nlargest(top_k_bs, 'cosine_sim').reset_index(drop=True)
                    print(f"    [{label}] Trimmed B-S to top {top_k_bs:,} by cosine_sim")
                    gc.collect()
            else:
                df_bs = pd.DataFrame(columns=['id1', 'id2', 'cosine_sim', 'pair_type'])

            # Free book embeddings now that both scans are done
            del book_embs_n
            gc.collect()

            return pd.concat([df_bb, df_bs], ignore_index=True)

        # --- Run over premises and conclusions separately ---
        print(f"\n=== Generating filtered premise-premise pairs ===")
        pairs_p = _filtered_pairs_within_df(df_p, "Premises")
        gc.collect()

        print(f"\n=== Generating filtered conclusion-conclusion pairs ===")
        pairs_sc = _filtered_pairs_within_df(df_sc, "Conclusions")
        gc.collect()

        # Combine (only id1, id2, cosine_sim, pair_type — NO text yet)
        all_pairs = pd.concat([pairs_p, pairs_sc], ignore_index=True)
        del pairs_p, pairs_sc
        gc.collect()

        # --- Final sampling (before adding heavy text columns) ---
        rng = np.random.default_rng(random_seed)
        mask_bb = all_pairs['pair_type'] == 'BB'
        mask_bs = all_pairs['pair_type'] == 'BS'

        if sample_n_bb is not None and mask_bb.sum() > sample_n_bb:
            bb_idx = rng.choice(np.where(mask_bb)[0], size=sample_n_bb, replace=False)
            print(f"Randomly sampled {sample_n_bb:,} B-B pairs from {mask_bb.sum():,}")
        else:
            bb_idx = np.where(mask_bb)[0]

        if top_k_bs is not None and mask_bs.sum() > top_k_bs:
            # Keep top_k_bs best BS pairs — use argpartition (O(n)) not nlargest (O(n log k))
            bs_indices = np.where(mask_bs)[0]
            bs_sims = all_pairs.loc[bs_indices, 'cosine_sim'].to_numpy()
            topk_pos = np.argpartition(bs_sims, -top_k_bs)[-top_k_bs:]
            bs_topk_idx = bs_indices[topk_pos]
            del bs_sims, topk_pos
        else:
            bs_topk_idx = np.where(mask_bs)[0]

        if sample_n_bs is not None and len(bs_topk_idx) > sample_n_bs:
            bs_idx = rng.choice(bs_topk_idx, size=sample_n_bs, replace=False)
            print(f"Randomly sampled {sample_n_bs:,} B-S pairs from top {len(bs_topk_idx):,}")
        else:
            bs_idx = bs_topk_idx

        keep_idx = np.concatenate([bb_idx, bs_idx])
        del bb_idx, bs_idx, bs_topk_idx, mask_bb, mask_bs
        gc.collect()

        # Slice to final small DataFrame, then free the big one
        all_pairs = all_pairs.iloc[keep_idx].reset_index(drop=True)
        del keep_idx
        gc.collect()

        # Add text columns ONLY to the small sampled result
        all_pairs['text1'] = all_pairs['id1'].map(id_to_text)
        all_pairs['text2'] = all_pairs['id2'].map(id_to_text)

        n_bb = (all_pairs['pair_type'] == 'BB').sum()
        n_bs = (all_pairs['pair_type'] == 'BS').sum()
        print(f"\n=== SUMMARY (threshold-filtered) ===")
        print(f"Book-Book pairs:    {n_bb:,}")
        print(f"Book-Speech pairs:  {n_bs:,}")
        print(f"Total pairs:        {len(all_pairs):,}")

        # Filter out already-labeled pairs
        if exclude_labeled_csv:
            all_pairs = filter_already_labeled(all_pairs, labeled_csv=exclude_labeled_csv)

        return all_pairs

    # ================================================================
    # ORIGINAL PATH  (random sampling, no thresholds)
    # ================================================================
    def create_pairs_within_df(
        df: pd.DataFrame,
        id_col: str,
        target_sample_size: int = None,
        group_label: str = '',
    ) -> pd.DataFrame:
        """Create randomly sampled pairwise combinations within a dataframe,
        excluding Speech-Speech.  Supports cross-round seen-index tracking
        via ``seen_indices_dir`` (captured from the outer scope).

        Parameters
        ----------
        df : pd.DataFrame
            Source rows (premises or conclusions).
        id_col : str
            Column with sentence IDs.
        target_sample_size : int or None
            How many pairs to sample.  ``None`` → generate all.
        group_label : str
            E.g. ``'premises'`` or ``'conclusions'``.  Used to
            namespace the ``.npy`` seen-index files.
        """
        ids = df[id_col].tolist()
        n = len(ids)

        # Categorize IDs by source type
        book_indices = [i for i, id_ in enumerate(ids) if get_source_type(id_) == 'B']
        speech_indices = [i for i, id_ in enumerate(ids) if get_source_type(id_) == 'S']

        n_book = len(book_indices)
        n_speech = len(speech_indices)

        book_book_max = n_book * (n_book - 1) // 2
        book_speech_max = n_book * n_speech
        max_possible_pairs = book_book_max + book_speech_max

        print(f"  Dataset size: {n} ({n_book} books, {n_speech} speeches)")
        print(f"  Max possible valid pairs: {max_possible_pairs:,}")

        # Vectorised ID arrays for fast indexing
        book_id_arr = np.array([ids[i] for i in book_indices])
        speech_id_arr = (np.array([ids[i] for i in speech_indices])
                         if speech_indices else np.array([], dtype=object))
        rng = np.random.default_rng(random_seed)

        # --- Load previously seen indices (if tracking enabled) ---
        seen_bb = np.array([], dtype=np.int64)
        seen_bs = np.array([], dtype=np.int64)
        if seen_indices_dir and group_label:
            bb_path = os.path.join(seen_indices_dir, f"seen_{group_label}_bb.npy")
            bs_path = os.path.join(seen_indices_dir, f"seen_{group_label}_bs.npy")
            seen_bb = _load_seen(bb_path)
            seen_bs = _load_seen(bs_path)
            bb_avail = book_book_max - len(seen_bb)
            bs_avail = book_speech_max - len(seen_bs)
            print(f"  Seen-index tracking: {len(seen_bb):,} BB + "
                  f"{len(seen_bs):,} BS already consumed "
                  f"({bb_avail:,} BB + {bs_avail:,} BS remaining)")

        # ---- helper: convert BB linear indices → (i, j) ----
        def _lin_to_bb(lin_idx):
            nn = n_book
            ii = (nn - 0.5 - np.sqrt((nn - 0.5)**2 - 2.0 * lin_idx)).astype(np.int64)
            jj = (lin_idx - ii * (2 * nn - ii - 1) // 2 + ii + 1).astype(np.int64)
            return ii, jj

        if target_sample_size is None or target_sample_size >= max_possible_pairs:
            # ---- Generate ALL valid pairs (vectorised) ----
            # When seen-index tracking is on, exclude previously consumed
            print(f"  Generating all {max_possible_pairs:,} pairs...")
            parts = []
            new_bb_idx = np.array([], dtype=np.int64)
            new_bs_idx = np.array([], dtype=np.int64)

            if n_book >= 2:
                if len(seen_bb) > 0:
                    all_bb = np.arange(book_book_max, dtype=np.int64)
                    mask = np.ones(book_book_max, dtype=bool)
                    mask[seen_bb[seen_bb < book_book_max]] = False
                    fresh_bb = all_bb[mask]
                    del all_bb, mask
                else:
                    fresh_bb = np.arange(book_book_max, dtype=np.int64)
                ii, jj = _lin_to_bb(fresh_bb)
                new_bb_idx = fresh_bb
                parts.append(pd.DataFrame({'id1': book_id_arr[ii], 'id2': book_id_arr[jj]}))
                del ii, jj, fresh_bb

            if n_book > 0 and n_speech > 0:
                if len(seen_bs) > 0:
                    all_bs = np.arange(book_speech_max, dtype=np.int64)
                    mask = np.ones(book_speech_max, dtype=bool)
                    mask[seen_bs[seen_bs < book_speech_max]] = False
                    fresh_bs = all_bs[mask]
                    del all_bs, mask
                else:
                    fresh_bs = np.arange(book_speech_max, dtype=np.int64)
                b_idx = fresh_bs // n_speech
                s_idx = fresh_bs % n_speech
                new_bs_idx = fresh_bs
                parts.append(pd.DataFrame({'id1': book_id_arr[b_idx],
                                           'id2': speech_id_arr[s_idx]}))
                del b_idx, s_idx, fresh_bs

            # Persist newly consumed indices
            if seen_indices_dir and group_label:
                _save_seen(bb_path, np.union1d(seen_bb, new_bb_idx))
                _save_seen(bs_path, np.union1d(seen_bs, new_bs_idx))

            return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['id1', 'id2'])
        else:
            # ---- Vectorised random sampling (scales to tens of millions) ----
            print(f"  Sampling {target_sample_size:,} pairs from {max_possible_pairs:,} possible...")

            # Proportional allocation (accounting for already-consumed)
            bb_avail = book_book_max - len(seen_bb)
            bs_avail = book_speech_max - len(seen_bs)
            total_avail = bb_avail + bs_avail

            if total_avail <= 0:
                print(f"  ⚠ All pairs already consumed in previous rounds!")
                return pd.DataFrame(columns=['id1', 'id2'])

            target_sample_size = min(target_sample_size, total_avail)

            if total_avail > 0:
                book_book_target = int(target_sample_size * (bb_avail / total_avail))
                book_speech_target = target_sample_size - book_book_target
            else:
                book_book_target = book_speech_target = 0

            # Clamp each target to its available count
            book_book_target = min(book_book_target, bb_avail)
            book_speech_target = min(book_speech_target, bs_avail)

            # If one segment was clamped, give the surplus to the other
            total_target = book_book_target + book_speech_target
            if total_target < target_sample_size:
                deficit = target_sample_size - total_target
                if book_book_target < bb_avail:
                    extra = min(deficit, bb_avail - book_book_target)
                    book_book_target += extra
                    deficit -= extra
                if deficit > 0 and book_speech_target < bs_avail:
                    book_speech_target += min(deficit, bs_avail - book_speech_target)

            parts = []
            new_bb_idx = np.array([], dtype=np.int64)
            new_bs_idx = np.array([], dtype=np.int64)

            # --- BB sampling ---
            if book_book_target > 0 and n_book >= 2:
                bb_target = min(book_book_target, bb_avail)
                if bb_target >= bb_avail and len(seen_bb) > 0:
                    # Take ALL remaining BB pairs
                    all_bb = np.arange(book_book_max, dtype=np.int64)
                    mask = np.ones(book_book_max, dtype=bool)
                    mask[seen_bb[seen_bb < book_book_max]] = False
                    lin_idx = all_bb[mask]
                    del all_bb, mask
                elif bb_target >= book_book_max:
                    # No seen indices; take all
                    lin_idx = np.arange(book_book_max, dtype=np.int64)
                else:
                    # Sample, excluding seen
                    lin_idx = _sample_excluding(rng, book_book_max, seen_bb, bb_target)

                ii, jj = _lin_to_bb(lin_idx)
                new_bb_idx = lin_idx
                parts.append(pd.DataFrame({'id1': book_id_arr[ii], 'id2': book_id_arr[jj]}))
                del ii, jj, lin_idx
                print(f"    BB: sampled {len(new_bb_idx):,} pairs "
                      f"(seen {len(seen_bb):,}, avail {bb_avail:,})")

            # --- BS sampling ---
            if book_speech_target > 0 and n_book > 0 and n_speech > 0:
                bs_target = min(book_speech_target, bs_avail)
                if bs_target >= bs_avail and len(seen_bs) > 0:
                    all_bs = np.arange(book_speech_max, dtype=np.int64)
                    mask = np.ones(book_speech_max, dtype=bool)
                    mask[seen_bs[seen_bs < book_speech_max]] = False
                    lin_idx = all_bs[mask]
                    del all_bs, mask
                elif bs_target >= book_speech_max:
                    lin_idx = np.arange(book_speech_max, dtype=np.int64)
                else:
                    lin_idx = _sample_excluding(rng, book_speech_max, seen_bs, bs_target)

                b_idx = lin_idx // n_speech
                s_idx = lin_idx % n_speech
                new_bs_idx = lin_idx
                parts.append(pd.DataFrame({'id1': book_id_arr[b_idx],
                                           'id2': speech_id_arr[s_idx]}))
                del lin_idx, b_idx, s_idx
                print(f"    BS: sampled {len(new_bs_idx):,} pairs "
                      f"(seen {len(seen_bs):,}, avail {bs_avail:,})")

            # Persist newly consumed indices
            if seen_indices_dir and group_label:
                _save_seen(bb_path, np.union1d(seen_bb, new_bb_idx))
                _save_seen(bs_path, np.union1d(seen_bs, new_bs_idx))

            result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['id1', 'id2'])
            print(f"  Generated {len(result):,} unique pairs")
            return result
    
    # Allocate sampling budget proportionally
    n_p = len(df_p)
    n_sc = len(df_sc)
    
    # Estimate proportions of each type
    total_weight = n_p + n_sc
    target_pairs_p = int(max_pairs * (n_p / total_weight))
    target_pairs_sc = max_pairs - target_pairs_p
    
    print(f"\n=== Generating premise-premise pairs ===")
    pairs_p = create_pairs_within_df(df_p, id_col, target_sample_size=target_pairs_p,
                                     group_label='premises')

    print(f"\n=== Generating conclusion-conclusion pairs ===")
    pairs_sc = create_pairs_within_df(df_sc, id_col, target_sample_size=target_pairs_sc,
                                      group_label='conclusions')

    # Combine both sets of pairs (id1, id2 only — no text columns)
    all_pairs = pd.concat([pairs_p, pairs_sc], ignore_index=True)

    print(f"\n=== SUMMARY ===")
    print(f"Premise-premise pairs: {len(pairs_p):,}")
    print(f"Conclusion-conclusion pairs: {len(pairs_sc):,}")
    print(f"Total valid pairs: {len(all_pairs):,}")
    print(f"Columns: {list(all_pairs.columns)}")

    # Filter out already-labeled pairs
    if exclude_labeled_csv:
        all_pairs = filter_already_labeled(all_pairs, labeled_csv=exclude_labeled_csv)

    return all_pairs


def generate_valid_pairs_by_type(
    df_p: pd.DataFrame,
    df_sc: pd.DataFrame,
    n: int,
    id_col: str = 'sentence_id',
    text_col: str = 'sentence',
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate N random Book-Book pairs and N random Book-Speech pairs
    from premise and conclusion dataframes.

    Rules (same as generate_valid_pairs):
    - Premise-Conclusion cross-pairs: NOT allowed
    - Speech-Speech pairs: NOT allowed
    - Book-Book pairs: allowed (returned in first df)
    - Book-Speech pairs: allowed (returned in second df)

    Pairs are drawn from both df_p (premises) and df_sc (conclusions),
    but premise and conclusion sentences are never mixed in the same pair.

    ID format expected: [B/S]XXXXX[p/sc]
    - First character: B (Book) or S (Speech)
    - Last 1-2 characters: p (premise) or sc (conclusion)

    Parameters
    ----------
    df_p : pd.DataFrame
        Dataframe containing premise sentences with an ID column.
    df_sc : pd.DataFrame
        Dataframe containing conclusion sentences with an ID column.
    n : int
        Number of pairs to generate for each type (B-B and B-S).
    id_col : str, default='sentence_id'
        Name of the column containing sentence IDs.
    text_col : str, default='sentence'
        Name of the column containing the text/sentence content.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Tuple of (df_bb, df_bs):
        - df_bb: DataFrame with N Book-Book pairs, columns ['id1', 'id2', 'text1', 'text2']
        - df_bs: DataFrame with N Book-Speech pairs, columns ['id1', 'id2', 'text1', 'text2']
    """
    rng = np.random.default_rng(random_seed)

    def get_source_type(text_id: str) -> str:
        if isinstance(text_id, str) and len(text_id) > 0:
            return text_id[0].upper()
        return None

    # Combine both dataframes, keeping track of their type suffix (p / sc)
    df_all = pd.concat([df_p, df_sc]).drop_duplicates(subset=id_col)
    id_to_text = dict(zip(df_all[id_col], df_all[text_col]))

    # Categorise IDs within each sentence-type group (premise / conclusion)
    # so we never mix premises with conclusions in a pair.
    groups = []  # list of (book_ids, speech_ids) per sentence-type
    for df_src in [df_p, df_sc]:
        ids = df_src[id_col].tolist()
        book_ids = [x for x in ids if get_source_type(x) == 'B']
        speech_ids = [x for x in ids if get_source_type(x) == 'S']
        groups.append((book_ids, speech_ids))

    # ---- Book-Book pairs ----
    print(f"\n=== Generating {n:,} Book-Book pairs ===")
    bb_pairs_set: set = set()
    bb_pairs_list = []

    # Build pool of Book-Book candidate generators per group
    bb_pools = []
    for book_ids, _ in groups:
        if len(book_ids) >= 2:
            bb_pools.append(book_ids)

    if not bb_pools:
        raise ValueError("Not enough Book IDs to form Book-Book pairs.")

    max_attempts = n * 10
    attempts = 0
    while len(bb_pairs_list) < n and attempts < max_attempts:
        # Pick a random group
        pool = bb_pools[rng.integers(len(bb_pools))]
        i, j = rng.choice(len(pool), size=2, replace=False)
        a, b = pool[i], pool[j]
        key = (min(a, b), max(a, b))
        if key not in bb_pairs_set:
            bb_pairs_set.add(key)
            bb_pairs_list.append({'id1': key[0], 'id2': key[1]})
        attempts += 1

    df_bb = pd.DataFrame(bb_pairs_list)
    df_bb['text1'] = df_bb['id1'].map(id_to_text)
    df_bb['text2'] = df_bb['id2'].map(id_to_text)
    print(f"  Generated {len(df_bb):,} Book-Book pairs")

    # ---- Book-Speech pairs ----
    print(f"\n=== Generating {n:,} Book-Speech pairs ===")
    bs_pairs_set: set = set()
    bs_pairs_list = []

    # Build pool of (book_ids, speech_ids) candidate generators per group
    bs_pools = []
    for book_ids, speech_ids in groups:
        if len(book_ids) > 0 and len(speech_ids) > 0:
            bs_pools.append((book_ids, speech_ids))

    if not bs_pools:
        raise ValueError("Not enough Book + Speech IDs to form Book-Speech pairs.")

    attempts = 0
    while len(bs_pairs_list) < n and attempts < max_attempts:
        pool_b, pool_s = bs_pools[rng.integers(len(bs_pools))]
        a = pool_b[rng.integers(len(pool_b))]
        b = pool_s[rng.integers(len(pool_s))]
        key = (min(a, b), max(a, b))
        if key not in bs_pairs_set:
            bs_pairs_set.add(key)
            bs_pairs_list.append({'id1': a, 'id2': b})
        attempts += 1

    df_bs = pd.DataFrame(bs_pairs_list)
    df_bs['text1'] = df_bs['id1'].map(id_to_text)
    df_bs['text2'] = df_bs['id2'].map(id_to_text)
    print(f"  Generated {len(df_bs):,} Book-Speech pairs")

    print(f"\n=== SUMMARY ===")
    print(f"Book-Book pairs:   {len(df_bb):,}")
    print(f"Book-Speech pairs: {len(df_bs):,}")

    return df_bb, df_bs


def setminus(
    df_big: pd.DataFrame,
    df_small: pd.DataFrame,
    id_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Perform set difference operation: df_big minus df_small
    
    Returns rows from df_big whose (id1, id2) pairs are not present in df_small.
    
    Parameters
    ----------
    df_big : pd.DataFrame
        The larger dataframe to subtract from.
    df_small : pd.DataFrame
        The smaller dataframe containing pairs to remove.
    id_cols : List[str], default=['id1', 'id2']
        The column names to use for matching pairs.
    
    Returns
    -------
    pd.DataFrame
        Rows from df_big that don't have matching id pairs in df_small.
    
    Examples
    --------
    >>> df_big = pd.DataFrame({'id1': ['A', 'B', 'C'], 'id2': ['X', 'Y', 'Z'], 'text1': [...], 'text2': [...]})
    >>> df_small = pd.DataFrame({'id1': ['A'], 'id2': ['X'], 'text1': [...], 'text2': [...], 'verdict': [1]})
    >>> result = setminus(df_big, df_small)
    >>> # result contains rows with (B,Y) and (C,Z) but not (A,X)
    """
    if id_cols is None:
        id_cols = ['id1', 'id2']

    c1, c2 = id_cols[0], id_cols[1]

    # Set-based anti-join: avoids creating a full merge copy of df_big.
    # df_small is always tiny (≤ 10 k rows), so the set lookup is instant.
    # Peak extra memory = 75 MB boolean mask + ~32 MB per 2 M-row chunk.
    remove_set = set(
        zip(df_small[c1].astype(str), df_small[c2].astype(str))
    )

    n = len(df_big)
    mask = np.ones(n, dtype=bool)
    chunk_sz = 2_000_000
    for lo in range(0, n, chunk_sz):
        hi = min(lo + chunk_sz, n)
        ids1 = df_big[c1].iloc[lo:hi].astype(str).tolist()
        ids2 = df_big[c2].iloc[lo:hi].astype(str).tolist()
        for j in range(hi - lo):
            if (ids1[j], ids2[j]) in remove_set:
                mask[lo + j] = False
        del ids1, ids2

    result = df_big[mask]
    result.reset_index(drop=True, inplace=True)

    print(f"Set difference: {len(df_big):,} - {len(df_small):,} = {len(result):,} rows")

    return result


def merge_pairwise_texts(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df1_cols: List[str],
    df2_cols: List[str],
) -> pd.DataFrame:
    """
    Merge a text dataframe (df1) with a pairwise dataframe (df2) to obtain
    (id1, id2, text1, text2, verdict), where 'verdict' is optional.
    
    Filters pairs based on ID patterns (e.g., BXXXXXXXsc, SXXXXXXXp):
    - First character: B (Book) or S (Speech)
    - Last 1-2 characters: sc (conclusion) or p (premise)
    
    Filtering rules:
    1. Validates ID format (B/S prefix + variable middle + sc/p suffix)
    2. Books-Books allowed, Books-Speech allowed, Speech-Speech NOT allowed
    3. Premise-Premise allowed, Conclusion-Conclusion allowed, Premise-Conclusion NOT allowed

    Parameters
    ----------
    df1 : pd.DataFrame
        DataFrame containing unique ids and their corresponding texts.
    df2 : pd.DataFrame
        DataFrame containing pairs of ids and optionally a verdict column.
    df1_cols : list of str
        Column names in df1 in the order [id_col, text_col].
        Example: ['id', 'text'].
    df2_cols : list of str
        Column names in df2. Two allowed patterns:
            - [id1_col, id2_col]
            - [id1_col, id2_col, verdict_col]

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'id1' : str
        - 'id2' : str
        - 'text1' : str (text corresponding to id1)
        - 'text2' : str (text corresponding to id2)
        - 'verdict' : verdict if provided in df2, otherwise NaN

    Notes
    -----
    - If df2_cols has length 2, the output 'verdict' column is created and
      filled with NaN.
    - If an id in df2 does not exist in df1, the corresponding text will be NaN.
    - Pairs violating comparison rules are filtered out.
    """
    # Unpack df1 columns
    if len(df1_cols) != 2:
        raise ValueError("df1_cols must have length 2: [id_col, text_col].")
    id_col_df1, text_col_df1 = df1_cols

    # Handle df2 columns (with or without verdict)
    if len(df2_cols) == 2:
        id1_col_df2, id2_col_df2 = df2_cols
        verdict_col_df2 = None
    elif len(df2_cols) == 3:
        id1_col_df2, id2_col_df2, verdict_col_df2 = df2_cols
    else:
        raise ValueError(
            "df2_cols must have length 2 ([id1_col, id2_col]) "
            "or 3 ([id1_col, id2_col, verdict_col])."
        )

    # Reduce to needed columns
    df1_reduced = df1[[id_col_df1, text_col_df1]].copy()
    df2_reduced = df2[[id1_col_df2, id2_col_df2] + ([verdict_col_df2] if verdict_col_df2 else [])].copy()

    # Rename id columns in df2 to standard names first for filtering
    df2_reduced = df2_reduced.rename(
        columns={id1_col_df2: "id1", id2_col_df2: "id2"}
    )
    
    # Filter pairs based on comparison rules (vectorised for millions of rows)
    if len(df2_reduced) == 0:
        print(f"Warning: df2 is empty (0 pairs). Returning empty result.")
        return pd.DataFrame(columns=["id1", "id2", "text1", "text2", "verdict"])

    id1_str = df2_reduced['id1'].astype(str)
    id2_str = df2_reduced['id2'].astype(str)
    src1 = id1_str.str[0].str.upper()
    src2 = id2_str.str[0].str.upper()
    suf1 = np.where(id1_str.str.endswith('sc'), 'sc',
                    np.where(id1_str.str.endswith('p'), 'p', ''))
    suf2 = np.where(id2_str.str.endswith('sc'), 'sc',
                    np.where(id2_str.str.endswith('p'), 'p', ''))
    valid_mask = (
        (suf1 != '') & (suf2 != '') &        # valid format
        ~((src1 == 'S') & (src2 == 'S')) &    # no Speech-Speech
        (suf1 == suf2)                        # same clause type
    )
    df2_filtered = df2_reduced[valid_mask].copy()

    if len(df2_filtered) == 0:
        print(f"Warning: All {len(df2_reduced)} pairs were filtered out by comparison rules.")
        return pd.DataFrame(columns=["id1", "id2", "text1", "text2", "verdict"])
    elif len(df2_filtered) < len(df2_reduced):
        print(f"Filtered {len(df2_reduced) - len(df2_filtered)} pairs (kept {len(df2_filtered)}).")

    # Prepare df1 for merging on id1 and id2
    df1_for_id1 = df1_reduced.rename(
        columns={id_col_df1: "id1", text_col_df1: "text1"}
    )
    df1_for_id2 = df1_reduced.rename(
        columns={id_col_df1: "id2", text_col_df1: "text2"}
    )

    # Attach text1 (for id1)
    merged = df2_filtered.merge(df1_for_id1, on="id1", how="left")

    # Attach text2 (for id2)
    merged = merged.merge(df1_for_id2, on="id2", how="left")

    # Standardize/construct verdict column
    if verdict_col_df2 is not None:
        merged = merged.rename(columns={verdict_col_df2: "verdict"})
    else:
        merged["verdict"] = np.nan

    # Reorder columns
    return merged[["id1", "id2", "text1", "text2", "verdict"]]

def create_embedding_cache(
    df_texts: pd.DataFrame,
    id_col: str,
    text_col: str,
    model_name: str = "BAAI/bge-large-en-v1.5",
    batch_size: int = 128,
    show_progress_bar: bool = True
) -> Dict[str, np.ndarray]:
    """
    Pre-compute embeddings for all unique texts and return a cache dictionary.
    This is the KEY OPTIMIZATION - embed once, reuse everywhere.

    Parameters
    ----------
    df_texts : pd.DataFrame
        DataFrame containing unique IDs and their texts.
    id_col : str
        Name of the ID column.
    text_col : str
        Name of the text column.
    model_name : str, default "BAAI/bge-large-en-v1.5"
        SentenceTransformer model to use.
    batch_size : int, default 128
        Batch size for encoding.
    show_progress_bar : bool, default True
        Whether to show progress bar during encoding.

    Returns
    -------
    dict
        Dictionary mapping {id: embedding_vector} for all texts.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model.half()

    # Get unique id-text pairs (in case of duplicates)
    df_unique = df_texts[[id_col, text_col]].drop_duplicates(subset=[id_col])
    
    print(f"\n>>> Creating Embedding Cache")
    print(f"Model: {model_name}")
    print(f"Total unique texts: {len(df_unique)}")
    print(f"Device: {device}")
    
    # Encode all texts
    embeddings = model.encode(
        df_unique[text_col].tolist(),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_tensor=False  # Return numpy arrays for efficiency
    )
    
    # Create dictionary mapping id -> embedding
    embedding_cache = dict(zip(
        df_unique[id_col].astype(str),  # Convert IDs to string for consistency
        embeddings
    ))
    
    # Clean up
    del model, embeddings
    torch.cuda.empty_cache()
    
    print(f"✓ Embedding cache created: {len(embedding_cache)} entries")
    print(f"Embedding dimension: {list(embedding_cache.values())[0].shape[0]}\n")
    
    return embedding_cache

def add_embeddings_from_cache(
    df: pd.DataFrame,
    embedding_cache: Dict[str, np.ndarray],
    id_col1: str,
    id_col2: str,
    emb_col1: str = "emb1",
    emb_col2: str = "emb2"
) -> pd.DataFrame:
    """
    Add embedding columns to a pairwise dataframe by looking them up in the cache.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with two ID columns.
    embedding_cache : dict
        Dictionary mapping {id: embedding_vector}.
    id_col1, id_col2 : str
        Column names for the two IDs.
    emb_col1, emb_col2 : str
        Names for the new embedding columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with two additional embedding columns.
    """
    df_out = df.copy()
    
    # Map IDs to embeddings (handle missing IDs gracefully)
    df_out[emb_col1] = df_out[id_col1].astype(str).map(embedding_cache)
    df_out[emb_col2] = df_out[id_col2].astype(str).map(embedding_cache)
    
    # Check for missing embeddings
    missing1 = df_out[emb_col1].isna().sum()
    missing2 = df_out[emb_col2].isna().sum()
    
    if missing1 > 0 or missing2 > 0:
        print(f"Warning: {missing1} missing embeddings for {id_col1}, {missing2} for {id_col2}")
    
    return df_out


def add_cosine_similarity_from_embeddings(
    df: pd.DataFrame,
    emb_col1: str,
    emb_col2: str,
    new_col: str = "cosine_sim"
) -> pd.DataFrame:
    """
    Compute row-wise cosine similarity between two embedding columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing two columns with vector embeddings.
    emb_col1 : str
        Name of the column holding the first embedding (array-like per row).
    emb_col2 : str
        Name of the column holding the second embedding (array-like per row).
    new_col : str, default "cosine_sim"
        Name of the new column with cosine similarities.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with an additional `new_col` containing
        the cosine similarity between `emb_col1` and `emb_col2` for each row.
    """
    df_out = df.copy()
    
    # Filter out rows with missing embeddings
    mask_valid = df_out[emb_col1].notna() & df_out[emb_col2].notna()
    
    if not mask_valid.any():
        print("Warning: No valid embedding pairs found.")
        df_out[new_col] = np.nan
        return df_out
    
    # Stack embeddings into 2D arrays (n_samples, dim)
    emb1 = np.stack(df_out.loc[mask_valid, emb_col1].to_numpy())
    emb2 = np.stack(df_out.loc[mask_valid, emb_col2].to_numpy())

    # Compute numerator: dot product row-wise
    numer = np.sum(emb1 * emb2, axis=1)

    # Compute norms
    norm1 = np.linalg.norm(emb1, axis=1)
    norm2 = np.linalg.norm(emb2, axis=1)

    # Avoid division by zero
    denom = norm1 * norm2
    # Use np.where to handle zero norms safely
    cos_sim = np.where(denom > 0, numer / denom, 0.0)

    # Assign scores only to valid rows
    df_out.loc[mask_valid, new_col] = cos_sim
    df_out.loc[~mask_valid, new_col] = np.nan
    
    return df_out

# ================================================================
# MEMORY-EFFICIENT HELPERS  (for 10M+ pair scale)
# ================================================================

def add_cosine_similarity_chunked(
    df: pd.DataFrame,
    embedding_cache: Dict[str, np.ndarray],
    id_col1: str,
    id_col2: str,
    new_col: str = "cosine_sim",
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    """
    Memory-efficient cosine-similarity computation from an embedding cache.

    Instead of storing one embedding-vector *per cell* in the DataFrame
    (which at 75 M rows × 768-dim × float32 × 2 columns ≈ 460 GB),
    this function processes the data in fixed-size chunks:

    1. For each chunk, extract only that chunk's IDs as strings.
    2. Look up the two embedding vectors for each pair by ID.
    3. Stack them into temporary numpy matrices (chunk_size × dim).
    4. Compute row-wise cosine similarity in bulk.
    5. Write the result directly into a float64 column.

    Peak extra memory ≈ 2 × chunk_size × dim × 4 bytes (e.g. 3 GB for
    500 k chunk with 768-dim embeddings) + the output float64 column
    (600 MB for 75 M rows).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *id_col1* and *id_col2*.
    embedding_cache : dict
        ``{id_string: np.ndarray}`` of pre-computed embeddings.
    id_col1, id_col2 : str
        Column names for the pair IDs.
    new_col : str, default "cosine_sim"
        Name of the output column.
    chunk_size : int, default 500_000
        Rows per chunk (tune to available RAM).

    Returns
    -------
    pd.DataFrame
        *df* with an additional column *new_col* (modified in-place;
        a reference is returned for convenience).
    """
    n = len(df)
    cos_sim = np.full(n, np.nan, dtype=np.float64)

    # Determine embedding dimension from the first cache entry
    sample_emb = next(iter(embedding_cache.values()))
    dim = sample_emb.shape[0]

    # Get the underlying pandas Series (NOT .astype(str).values which
    # would allocate a 75M-element numpy array of Python str objects ≈ 5GB)
    col1_series = df[id_col1]
    col2_series = df[id_col2]

    n_chunks = (n + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        lo = ci * chunk_size
        hi = min(lo + chunk_size, n)
        sz = hi - lo

        # Extract IDs for THIS chunk only via .iloc → small string list
        chunk_ids1 = col1_series.iloc[lo:hi].astype(str).tolist()
        chunk_ids2 = col2_series.iloc[lo:hi].astype(str).tolist()

        emb1 = np.empty((sz, dim), dtype=np.float32)
        emb2 = np.empty((sz, dim), dtype=np.float32)
        valid = np.ones(sz, dtype=bool)

        for j in range(sz):
            e1 = embedding_cache.get(chunk_ids1[j])
            e2 = embedding_cache.get(chunk_ids2[j])
            if e1 is None or e2 is None:
                valid[j] = False
                emb1[j] = 0.0
                emb2[j] = 0.0
            else:
                emb1[j] = e1
                emb2[j] = e2

        # Vectorised cosine similarity
        numer = np.sum(emb1 * emb2, axis=1)
        norm1 = np.linalg.norm(emb1, axis=1)
        norm2 = np.linalg.norm(emb2, axis=1)
        denom = norm1 * norm2
        chunk_sim = np.where(denom > 0, numer / denom, 0.0)
        chunk_sim[~valid] = np.nan
        cos_sim[lo:hi] = chunk_sim

        del emb1, emb2, chunk_ids1, chunk_ids2
        if (ci + 1) % 10 == 0 or ci == n_chunks - 1:
            print(f"  cosine-sim chunk {ci + 1}/{n_chunks} done")

    df[new_col] = cos_sim
    return df


def build_equiv_map(
    df_entailed: pd.DataFrame,
    id1_col: str = "id1",
    id2_col: str = "id2",
    include_self: bool = False,
) -> Dict[str, set]:
    """
    Build an equivalence map from observed entailed pairs.

    Parameters
    ----------
    df_entailed : pd.DataFrame
        Pairs that have been labelled YES (or equivalent).
    id1_col, id2_col : str
        Column names.
    include_self : bool, default False
        Whether to keep the ID itself in its own equivalence set.

    Returns
    -------
    dict[str, set[str]]
        ``{id: {equivalent_ids}}``
    """
    equiv: Dict[str, set] = defaultdict(set)
    pairs = df_entailed[[id1_col, id2_col]].dropna().astype(str)
    for a, b in pairs.itertuples(index=False, name=None):
        pair = {a, b}
        for x in pair:
            equiv[x].update(pair)
    if not include_self:
        for x in equiv:
            equiv[x].discard(x)
    return dict(equiv)


def add_alpha_vectorized(
    df: pd.DataFrame,
    equiv_map: Dict[str, set],
    id_col1: str = "id1",
    id_col2: str = "id2",
    new_col: str = "alpha",
    chunk_size: int = 2_000_000,
) -> pd.DataFrame:
    """
    Vectorised alpha-weight column — no ``apply()``, no ``df.copy()``.

    Rules (identical to ``alpha_weight``):
    - both IDs without equivalents → NaN
    - only id1 has equivalents     → 1.0
    - only id2 has equivalents     → 0.0
    - both have equivalents        → 0.5

    Memory cost: one float64 output array (≈8 bytes/row) + small chunk
    boolean arrays.  Never materialises full 75 M-element string arrays.
    """
    ids_with_equivs = frozenset(k for k, v in equiv_map.items() if v)
    n = len(df)
    alpha = np.full(n, np.nan, dtype=np.float64)

    col1 = df[id_col1]
    col2 = df[id_col2]

    n_chunks = (n + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        lo = ci * chunk_size
        hi = min(lo + chunk_size, n)

        h1 = col1.iloc[lo:hi].astype(str).isin(ids_with_equivs).values
        h2 = col2.iloc[lo:hi].astype(str).isin(ids_with_equivs).values

        alpha[lo:hi] = np.where(
            h1 & h2, 0.5,
            np.where(h1 & ~h2, 1.0,
            np.where(~h1 & h2, 0.0, np.nan))
        )
        del h1, h2

    df[new_col] = alpha
    return df


def build_equiv_pair_candidates_from_map(
    df: pd.DataFrame,
    equiv_map: Dict[str, set],
    id1_col: str = "id1",
    id2_col: str = "id2",
    chunk_size: int = 2_000_000,
) -> pd.DataFrame:
    """
    Memory-efficient replacement for
    ``add_equivalents_from_pairs`` → ``build_equiv_pair_candidates``.

    Instead of materialising two list-valued columns across *all* rows
    (≈ 8 GB of empty Python list objects for 75 M rows), this function:

    1. Processes the DataFrame in chunks to avoid creating 75 M-element
       string arrays.
    2. For each chunk, identifies the (small) subset of rows where at
       least one ID has equivalents.
    3. Expands only that subset into crossed pairs.

    Returns
    -------
    pd.DataFrame
        Columns ``['id1', 'id2']`` — the union of
        ``(id1 × equivalents_of(id2))`` and ``(id2 × equivalents_of(id1))``.
    """
    n = len(df)
    col1 = df[id1_col]
    col2 = df[id2_col]

    parts: List[pd.DataFrame] = []
    n_chunks = (n + chunk_size - 1) // chunk_size

    for ci in range(n_chunks):
        lo = ci * chunk_size
        hi = min(lo + chunk_size, n)

        s1 = col1.iloc[lo:hi].astype(str).tolist()
        s2 = col2.iloc[lo:hi].astype(str).tolist()

        rows: List[Tuple[str, str]] = []
        for j in range(hi - lo):
            i1, i2 = s1[j], s2[j]
            eq1 = equiv_map.get(i1)
            eq2 = equiv_map.get(i2)
            if eq2:
                for k in eq2:
                    rows.append((i1, k))
            if eq1:
                for k in eq1:
                    rows.append((i2, k))

        if rows:
            parts.append(pd.DataFrame(rows, columns=["id1", "id2"]))
        del s1, s2, rows

    if parts:
        out = pd.concat(parts, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["id1", "id2"])
    return out


def compute_neighbor_score_efficient(
    sigma_lookup: Dict[Tuple[str, str], float],
    df6: pd.DataFrame,
    equiv_map: Dict[str, set],
    id1_col: str = "id1",
    id2_col: str = "id2",
    cosim_col: str = "new_cos_sim_score",
    alpha_col: str = "alpha",
    new_col: str = "cos_sim_neighbor_score",
    chunk_size: int = 2_000_000,
) -> pd.DataFrame:
    """
    Memory-efficient neighbor-weighted score (replaces ``compute_neighbor_weighted_score``).

    Key optimisations
    -----------------
    * **Chunked ID extraction**: Never materialises 75 M-element string
      arrays; processes IDs in 2 M-row chunks.
    * **Short-circuit**: For the vast majority of rows where *both*
      equivalence sets are empty, score = σ(i, j) directly.
    * **No list columns**: Equivalences are looked up from *equiv_map*
      on the fly for the small subset of rows that need them.
    * **No df.copy()**: The result column is written in-place.
    """
    n = len(df6)
    scores = np.full(n, np.nan, dtype=np.float64)

    # The cosine-sim and alpha columns are plain float64 — safe to pull at once
    alphas = df6[alpha_col].values.astype(np.float64)
    sigma_ij_arr = df6[cosim_col].values.astype(np.float64)

    ids_with = frozenset(k for k, v in equiv_map.items() if v)

    col1 = df6[id1_col]
    col2 = df6[id2_col]

    # Collect indices of rows that need per-row neighbor computation
    neighbor_indices: List[int] = []

    n_chunks = (n + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        lo = ci * chunk_size
        hi = min(lo + chunk_size, n)

        chunk_s1 = col1.iloc[lo:hi].astype(str).tolist()
        chunk_s2 = col2.iloc[lo:hi].astype(str).tolist()

        for j in range(hi - lo):
            h1 = chunk_s1[j] in ids_with
            h2 = chunk_s2[j] in ids_with
            if h1 or h2:
                neighbor_indices.append(lo + j)
            else:
                # Bulk shortcut: no neighbours → score = σ_ij
                scores[lo + j] = sigma_ij_arr[lo + j]
        del chunk_s1, chunk_s2

    print(f"  neighbor score: {len(neighbor_indices):,} / {n:,} rows need "
          f"neighbour computation ({100 * len(neighbor_indices) / max(n, 1):.1f}%)")

    def _get_sigma(a: str, b: str) -> float:
        return sigma_lookup.get((a, b), np.nan)

    def _mean_sigma(anchor: str, others: set) -> float:
        vals = [_get_sigma(anchor, k) for k in others if k != anchor]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else 0.0

    # Only need string lookups for the (small) neighbor subset
    for idx in neighbor_indices:
        i = str(col1.iat[idx])
        j = str(col2.iat[idx])
        alpha = alphas[idx]
        sigma_ij = sigma_ij_arr[idx]

        if np.isnan(sigma_ij):
            continue  # stays NaN

        eq_i = equiv_map.get(i, set()) - {i}
        eq_j = equiv_map.get(j, set()) - {j}

        if len(eq_i) == 0 and len(eq_j) == 0:
            scores[idx] = sigma_ij
            continue

        if np.isnan(alpha):
            continue  # stays NaN

        term_i = alpha * _mean_sigma(j, eq_i) if (alpha != 0 and eq_i) else 0.0
        term_j = (1.0 - alpha) * _mean_sigma(i, eq_j) if (alpha != 1 and eq_j) else 0.0
        scores[idx] = sigma_ij * (term_i + term_j)

    df6[new_col] = scores
    return df6


def prepare_candidates_efficient(
    df_obs_ent: pd.DataFrame,
    df_predict: pd.DataFrame,
    df_clause: pd.DataFrame,
    obs_id_cols: Tuple[str, str] = ("id1", "id2"),
    predict_id_cols: Tuple[str, str] = ("id1", "id2"),
    clause_id_col: str = "sentence_id",
    clause_text_col: str = "sentence",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, set]]:
    """
    Memory-efficient replacement for the four-function chain:

        add_equivalents_from_pairs
        → add_alpha_weight_column
        → build_equiv_pair_candidates
        → merge_pairwise_texts

    Instead of storing Python list objects in every row of *df_predict*
    (≈ 8 GB overhead for 75 M rows), this function:

    1. Builds a compact ``equiv_map`` (dict of sets).
    2. Computes alpha weights with vectorised ``isin()`` (no ``apply()``).
    3. Generates crossed pairs directly from the map for only the
       relevant subset of rows.
    4. Merges clause texts only onto the (much smaller) crossed-pairs df.

    Parameters
    ----------
    df_obs_ent : pd.DataFrame
        Observed entailed pairs (verdict == YES).
    df_predict : pd.DataFrame
        Candidate pairs (the large pool, e.g. 75 M rows).
    df_clause : pd.DataFrame
        All clauses with *clause_id_col* and *clause_text_col*.
    obs_id_cols : tuple of str
        (id1, id2) column names in *df_obs_ent*.
    predict_id_cols : tuple of str
        (id1, id2) column names in *df_predict*.
    clause_id_col, clause_text_col : str
        Column names in *df_clause*.

    Returns
    -------
    (df_predict, df_crossed, equiv_map)
        * df_predict — same object with an ``alpha`` column added in-place.
        * df_crossed — crossed pairs with ``id1 id2 text1 text2``.
        * equiv_map — ``dict[str, set[str]]`` for later use.
    """
    # 1. Equivalence map
    equiv_map = build_equiv_map(
        df_obs_ent,
        id1_col=obs_id_cols[0],
        id2_col=obs_id_cols[1],
        include_self=False,
    )
    print(f"  equiv_map: {len(equiv_map)} IDs with equivalents")

    # 2. Alpha (vectorised, in-place)
    add_alpha_vectorized(
        df_predict,
        equiv_map,
        id_col1=predict_id_cols[0],
        id_col2=predict_id_cols[1],
    )

    # 3. Crossed pairs (from map, no list columns)
    df_crossed = build_equiv_pair_candidates_from_map(
        df_predict,
        equiv_map,
        id1_col=predict_id_cols[0],
        id2_col=predict_id_cols[1],
    )
    print(f"  crossed pairs: {len(df_crossed):,} rows")

    # 4. Merge texts onto crossed pairs only
    df_crossed = merge_pairwise_texts(
        df1=df_clause,
        df2=df_crossed,
        df1_cols=[clause_id_col, clause_text_col],
        df2_cols=["id1", "id2"],
    )

    return df_predict, df_crossed, equiv_map


def add_cosine_similarity_from_text(
    df: pd.DataFrame,
    text_col1: str,
    text_col2: str,
    model_name: str = "BAAI/bge-en-icl",
    new_col: str = "cosine_sim",
    batch_size: int = 128,  
    show_progress_bar: bool = True,
    embedding_cache: Optional[Dict[str, np.ndarray]] = None,
    id_col1: Optional[str] = None,
    id_col2: Optional[str] = None
) -> pd.DataFrame:
    """
    Encode two text columns with a SentenceTransformer model and compute
    row-wise cosine similarity between them in a vectorized way.
    
    NEW: Now supports using pre-computed embeddings from a cache for efficiency!

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing two text columns (or ID columns if using cache).
    text_col1 : str
        Name of the first text column.
    text_col2 : str
        Name of the second text column.
    model_name : str, default "BAAI/bge-en-icl"
        SentenceTransformer model to use (ignored if embedding_cache provided).
    new_col : str, default "cosine_sim"
        Name of the new column with cosine similarities.
    batch_size : int, default 128
        Batch size for encoding (ignored if embedding_cache provided).
    show_progress_bar : bool, default True
        Whether to show progress bar during encoding (ignored if embedding_cache provided).
    embedding_cache : dict, optional
        Pre-computed embeddings {id: vector}. If provided, uses cached embeddings.
    id_col1, id_col2 : str, optional
        ID columns to use for cache lookup (required if embedding_cache is provided).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with an additional column `new_col` containing
        cosine similarities between the embeddings of the two text columns.
    """
    # NEW: If cache provided, use it instead of re-encoding
    if embedding_cache is not None:
        if id_col1 is None or id_col2 is None:
            raise ValueError("When using embedding_cache, must provide id_col1 and id_col2")
        
        if len(df) == 0:
            df[new_col] = pd.Series(dtype=float)
            return df
        
        # For large DataFrames (>5M rows), use memory-efficient chunked path
        # to avoid storing 2 embedding-vector columns in the DataFrame.
        LARGE_THRESHOLD = 5_000_000
        if len(df) > LARGE_THRESHOLD:
            print(f"Using chunked cosine-similarity path ({len(df):,} rows)...")
            return add_cosine_similarity_chunked(
                df, embedding_cache, id_col1, id_col2,
                new_col=new_col, chunk_size=500_000,
            )

        print(f"Using pre-computed embeddings from cache...")
        
        # Add embeddings from cache
        df_with_emb = add_embeddings_from_cache(
            df, embedding_cache, id_col1, id_col2,
            emb_col1="_temp_emb1", emb_col2="_temp_emb2"
        )
        
        # Compute cosine similarity
        df_with_cos = add_cosine_similarity_from_embeddings(
            df_with_emb, "_temp_emb1", "_temp_emb2", new_col
        )
        
        # Remove temporary embedding columns
        df_out = df_with_cos.drop(columns=["_temp_emb1", "_temp_emb2"])
        
        return df_out
    
    # ORIGINAL: Encode on-the-fly (legacy path for backward compatibility)
    if len(df) == 0:
        df[new_col] = pd.Series(dtype=float)
        return df

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    #Saves some RAM/VRAM
    if device == "cuda":
        model.half() 

    print(f"Encoding unique sentences from {text_col1} and {text_col2}...")
    
    # 3. Optimized Encoding: Don't encode duplicates
    # We combine both columns, find unique strings, encode them, then map back
    unique_sentences = pd.concat([df[text_col1], df[text_col2]]).unique().tolist()
    
    embeddings_map = dict(zip(
        unique_sentences, 
        model.encode(
            unique_sentences, 
            batch_size=batch_size, 
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True
        )
    ))

    # 4. Map embeddings back to the dataframe columns
    emb1 = torch.stack([embeddings_map[s] for s in df[text_col1]])
    emb2 = torch.stack([embeddings_map[s] for s in df[text_col2]])

    # 5. Vectorized Cosine Similarity calculation (Row-wise)
    # util.cos_sim returns a matrix; we only want the diagonal (row-to-row)
    # We do it manually to be more memory efficient:
    cosine_sims = torch.nn.functional.cosine_similarity(emb1, emb2)

    df[new_col] = cosine_sims.cpu().numpy()
    
    # Clear VRAM cache
    del emb1, emb2, embeddings_map
    torch.cuda.empty_cache()

    return df

def add_cross_encoder_score(
    df: pd.DataFrame,
    text_col1: str,
    text_col2: str,
    model_name: str = "cross-encoder/nli-deberta-v3-base", # Very accurate NLI model
    new_col: str = "nli_score",
    batch_size: int = 64,
    show_progress_bar: bool = True
) -> pd.DataFrame:
    """
    Adds a column with NLI scores predicted by a Cross-Encoder model.
    Processes the pair (A,B) jointly and weights NLI similarity of A->B and B->A.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Cross-Encoder model: {model_name} on {device}...")
    # use_fast=False helps avoid tokenizer serialization errors with DeBERTa-v2/v3 models on some systems
    model = CrossEncoder(model_name, device=device, tokenizer_args={"use_fast": False})
    
    # Prepare pairs for both directions (A->B and B->A)
    pairs_ab = list(zip(df[text_col1], df[text_col2]))
    pairs_ba = list(zip(df[text_col2], df[text_col1]))
    
    print(f"Predicting NLI scores for {len(pairs_ab)} pairs (Bidirectional)...")
    
    # Predict returns logits. 
    # For 'cross-encoder/nli-deberta-v3-base', labels are: 0: Contradiction, 1: Entailment, 2: Neutral
    scores_ab = model.predict(pairs_ab, batch_size=batch_size, show_progress_bar=show_progress_bar)
    scores_ba = model.predict(pairs_ba, batch_size=batch_size, show_progress_bar=show_progress_bar)
    
    # Softmax to get probabilities
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
        
    probs_ab = softmax(scores_ab)
    probs_ba = softmax(scores_ba)
    
    # We take the probability of 'Entailment' (Label 1)
    entail_ab = probs_ab[:, 1]
    entail_ba = probs_ba[:, 1]
    
    # Combine scores (Product of probabilities for equivalence)
    df[new_col] = entail_ab * entail_ba
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return df

#SENDING TO LLM
def two_random_subsamples(
    df: pd.DataFrame,
    frac1: float,
    frac2: float,
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Draw two non-overlapping random subsamples from a DataFrame.

    The function uses a single random permutation of the DataFrame's indices
    and then splits that permutation into two disjoint subsets, corresponding
    to the requested fractions.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame to sample from.
    frac1 : float
        Fraction of rows to include in the first subsample (between 0 and 1).
    frac2 : float
        Fraction of rows to include in the second subsample (between 0 and 1).
    random_state : int or None, default None
        Seed for the random number generator (for reproducibility).
        If None, a random seed is used.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        A tuple (sub1, sub2) of two disjoint DataFrames.

    Raises
    ------
    ValueError
        If frac1 or frac2 are not in [0, 1], or if frac1 + frac2 > 1.

    Notes
    -----
    - Sampling is done *without replacement* and the two subsamples are
      guaranteed to have no overlapping rows.
    - The final sizes are:
        n1 = int(round(frac1 * n))
        n2 = int(round(frac2 * n))
      where n is len(df).
    """
    n = len(df)

    if not (0 <= frac1 <= 1) or not (0 <= frac2 <= 1):
        raise ValueError("frac1 and frac2 must be in [0, 1].")

    # Planned sizes - use floor to avoid rounding errors when frac1 + frac2 = 1
    n1 = int(frac1 * n)
    n2 = int(frac2 * n)
    
    # Ensure we don't exceed n (can happen due to rounding)
    if n1 + n2 > n:
        # Adjust n2 to fit
        n2 = n - n1

    rng = np.random.default_rng(random_state)
    permuted_idx = rng.permutation(df.index.to_numpy())

    idx1 = permuted_idx[:n1]
    idx2 = permuted_idx[n1:n1 + n2]

    sub1 = df.loc[idx1].copy()
    sub2 = df.loc[idx2].copy()

    return sub1, sub2

def add_equivalents_from_pairs(
    df3: pd.DataFrame,
    df4: pd.DataFrame,
    df3_cols: List[str],
    df4_cols: List[str],
    new_cols: Tuple[str, str] = ("equivalents1", "equivalents2"),
    include_self: bool = True
) -> pd.DataFrame:
    """
    Given:
        - df3: an equivalence-pairs dataframe with two ID columns.
        - df4: a dataframe with (id1, id2, cosine_sim, ...).

    Construct for every ID in df4:
        - a list of all IDs that co-occur with it in df3 (same row in df3),
          considering that the ID may appear in either df3 column.

    Parameters
    ----------
    df3 : pd.DataFrame
        DataFrame with exactly two ID columns indicating equivalence pairs.
    df4 : pd.DataFrame
        DataFrame with at least two ID columns; additional columns (e.g.
        cosine_sim) are preserved.
    df3_cols : list of str
        Column names in df3 in the order [id1_col_df3, id2_col_df3].
    df4_cols : list of str
        Column names in df4 in the order [id1_col_df4, id2_col_df4].
    new_cols : tuple of str, default ("equivalents1", "equivalents2")
        Names of the new columns to add to df4, corresponding to df4_cols.
    include_self : bool, default True
        If True, the ID itself will be included in its own equivalence list.
        If False, it will be removed from the list (only “other” equivalents).

    Returns
    -------
    pd.DataFrame
        A copy of df4 with two additional list-valued columns:
        - new_cols[0]: all IDs equivalent to df4[id1_col_df4]
        - new_cols[1]: all IDs equivalent to df4[id2_col_df4]

        Each cell in these new columns is a Python list of strings.
        If an ID does not appear in df3 at all, the corresponding list is [].

    Notes
    -----
    - Equivalence is defined only row-wise (no transitive closure). If you need
      full equivalence classes (connected components), build them separately.
    - The mapping from an ID to its equivalents is pre-computed once in
      O(n_rows_df3) time, and then applied to df4 via vectorized `Series.map`.
    """
    id1_df3, id2_df3 = df3_cols
    id1_df4, id2_df4 = df4_cols
    new_col1, new_col2 = new_cols

    # Build mapping: id -> set of "row-wise" equivalents from df3
    equiv_map = defaultdict(set)

    # Drop rows where either ID is missing; cast to string to standardize
    df3_pairs = df3[[id1_df3, id2_df3]].dropna().astype(str)

    # Complexity O(n_rows_df3); uses Python sets and dicts (fast enough for most cases)
    for a, b in df3_pairs.itertuples(index=False, name=None):
        pair = {a, b}
        for x in pair:
            equiv_map[x].update(pair)

    # Optionally remove the ID itself from its own list
    if not include_self:
        for x in equiv_map:
            equiv_map[x].discard(x)

    # Apply mapping to df4 (vectorized via Series.map), standardizing to string
    df4_out = df4.copy()
    df4_out[new_col1] = (
        df4_out[id1_df4].astype(str).map(lambda x: sorted(equiv_map.get(x, set())))
    )
    df4_out[new_col2] = (
        df4_out[id2_df4].astype(str).map(lambda x: sorted(equiv_map.get(x, set())))
    )

    return df4_out

def alpha_weight(list1: List[Any], list2: List[Any]) -> float:
    """
    Compute an "alpha weight" based on emptiness of two lists.

    Rules
    -----
    - If both lists are empty        -> return np.nan
    - If list1 is empty, list2 not   -> return 0.0
    - If list1 not empty, list2 is   -> return 1.0
    - If both lists are non-empty    -> return 0.5

    Parameters
    ----------
    list1 : list
        First list.
    list2 : list
        Second list.

    Returns
    -------
    float
        The alpha weight as defined above (or np.nan if both lists are empty).
    """
    is_empty1 = (len(list1) == 0)
    is_empty2 = (len(list2) == 0)

    if is_empty1 and is_empty2:
        return float(np.nan)
    if is_empty1 and not is_empty2:
        return 0.0
    if not is_empty1 and is_empty2:
        return 1.0
    # both non-empty
    return 0.5

def add_alpha_weight_column(
    df: pd.DataFrame,
    list_col1: str,
    list_col2: str,
    new_col: str = "alpha"
) -> pd.DataFrame:
    """
    Add an alpha-weight column to a DataFrame given two columns of lists.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing two columns whose entries are lists.
    list_col1 : str
        Name of the first list-valued column.
    list_col2 : str
        Name of the second list-valued column.
    new_col : str, default "alpha"
        Name of the new column to create.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with an additional column `new_col`
        containing alpha_weight(list1, list2) for each row.

    Notes
    -----
    - `None` or NaN in list columns are treated as empty lists.
    - The computation is row-wise (uses `DataFrame.apply`).
    """

    def _as_list(x: Any) -> list:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        # if it is something else (e.g. a scalar), treat as single-element list
        return [x]

    df_out = df.copy()
    df_out[new_col] = df_out.apply(
        lambda row: alpha_weight(
            _as_list(row[list_col1]),
            _as_list(row[list_col2])
        ),
        axis=1
    )
    return df_out

def build_equiv_pair_candidates(
    df: pd.DataFrame,
    id1_col: str = "id1",
    id2_col: str = "id2",
    equiv1_col: str = "equivalents1",
    equiv2_col: str = "equivalents2",
) -> pd.DataFrame:
    """
    From a dataframe with (id1, id2, equivalents1, equivalents2), construct
    a new dataframe whose rows are:

        - id1 × equivalents2  (pairs (id1, k) for k in equivalents2)
        - id2 × equivalents1  (pairs (id2, k) for k in equivalents1)

    and return them concatenated.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns id1, id2, equivalents1, equivalents2.
        The equivalents columns are expected to contain iterables (lists/tuples)
        or scalars (which will be treated as length-1 lists).
    id1_col, id2_col : str
        Column names for the two ids.
    equiv1_col, equiv2_col : str
        Column names for the equivalents of id1 and id2.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            - 'id1'
            - 'id2'
        containing all pairs from:
            (id1 × equivalents2) ∪ (id2 × equivalents1)

        Rows where the relevant equivalents list is empty are simply ignored.
    """
    df_norm = df.copy()

    # Ensure equivalents columns are lists
    df_norm[equiv1_col] = df_norm[equiv1_col].apply(_ensure_list)
    df_norm[equiv2_col] = df_norm[equiv2_col].apply(_ensure_list)

    # Part 1: id1 × equivalents2  -> (id1, k)
    part1 = (
        df_norm[[id1_col, equiv2_col]]
        .explode(equiv2_col)  # one row per element of equivalents2
        .dropna(subset=[equiv2_col])
        .rename(columns={id1_col: "id1", equiv2_col: "id2"})
    )

    # Part 2: id2 × equivalents1  -> (id2, k)
    part2 = (
        df_norm[[id2_col, equiv1_col]]
        .explode(equiv1_col)
        .dropna(subset=[equiv1_col])
        .rename(columns={id2_col: "id1", equiv1_col: "id2"})
    )

    # Concatenate both sets of pairs
    out = pd.concat(
        [part1[["id1", "id2"]], part2[["id1", "id2"]]],
        ignore_index=True
    )

    return out


# TRAINING BERT MODELS
def train_bert_model(
    df_train: pd.DataFrame,
    text_col1: str,
    text_col2: str,
    verdict_col: str,
    base_model_name: str = "princeton-nlp/sup-simcse-roberta-large",
    output_path: str = "./fine_tuned_simcse",
    positive_label: str = "YES",
    num_epochs: int = 4,
    batch_size: int = 64,  # Increased default for A100 (was 32)
    model_type: Literal["bi-encoder", "cross-encoder"] = "bi-encoder",
    # --- RCC OPTIMIZATIONS ---
    use_amp: bool = True,   # Enables FP16 (Mixed Precision)
    num_workers: int = 8    # Uses CPU cores to load data
):
    """
    Fine-tunes a Transformer model on labeled entailment pairs.
    Optimized for RCC Midway3 (A100 GPUs).
    """
    
    # 1. Filter for rows that actually have a verdict
    df_labeled = df_train.dropna(subset=[verdict_col])
    
    print(f"Training {model_type} on {len(df_labeled)} labeled pairs from {base_model_name}...")
    
    # 2. Convert DataFrame to InputExample list
    train_examples = []
    
    # Determine Label mappings
    # For Bi-Encoder: 1=Similar, 0=Dissimilar
    # For Cross-Encoder: 1=YES, 0=NO
    for _, row in df_labeled.iterrows():
        label = 1 if row[verdict_col] == positive_label else 0
        train_examples.append(
            InputExample(texts=[str(row[text_col1]), str(row[text_col2])], label=label)
        )

    # --- RCC OPTIMIZATION: Fast Data Loading ---
    # We create the loader once here since logic is similar for both
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=batch_size,
        num_workers=num_workers,  # <--- CRITICAL SPEEDUP
        pin_memory=True           # <--- CRITICAL SPEEDUP
    )

    if model_type == "bi-encoder":
        model = SentenceTransformer(base_model_name)
        train_loss = losses.ContrastiveLoss(model=model)
        
        print(f"Starting Bi-Encoder training with AMP={use_amp}...")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=100,
            show_progress_bar=True,
            optimizer_params={'lr': 2e-5},
            use_amp=use_amp  # <--- ENABLE FP16
        )
        
    elif model_type == "cross-encoder":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CrossEncoder(
            base_model_name, 
            num_labels=2, 
            device=device,
            tokenizer_args={"use_fast": False}
        )
        
        print(f"Starting Cross-Encoder training with AMP={use_amp}...")
        model.fit(
            train_dataloader=train_dataloader,
            epochs=num_epochs,
            warmup_steps=100,
            show_progress_bar=True,
            optimizer_params={'lr': 2e-5},
            use_amp=use_amp  # <--- ENABLE FP16
        )
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # 7. Save
    print(f"Saving fine-tuned model to {output_path}...")
    model.save(output_path)
    
    # Clean up VRAM
    del model
    torch.cuda.empty_cache()
    print("Training complete and VRAM cleared.")

def generate_new_bert_results(
    df: pd.DataFrame,
    text_col1: str,
    text_col2: str,
    model_path: str = "./fine_tuned_simcse",
    new_col: str = "bert_score",
    embedding_cache: Optional[Dict[str, np.ndarray]] = None,
    id_col1: Optional[str] = None,
    id_col2: Optional[str] = None
) -> pd.DataFrame:
    """
    Runs inference using the newly fine-tuned model saved at `model_path`.
    This is essentially a wrapper around add_cosine_similarity_from_text 
    but points to the local folder.
    
    NEW: Now supports using pre-computed embeddings from cache!
    """
    # Reuse the optimized inference function defined earlier
    return add_cosine_similarity_from_text(
        df=df,
        text_col1=text_col1,
        text_col2=text_col2,
        model_name=model_path, # Load from local folder
        new_col=new_col,
        batch_size=128, # Inference batch size can be slightly larger than training
        embedding_cache=embedding_cache,
        id_col1=id_col1,
        id_col2=id_col2
    )


# FEATURES
def _build_sigma_lookup_from_df5(
    df5: pd.DataFrame,
    id1_col: str = "id1",
    id2_col: str = "id2",
    cosim_col: str = "cosim",
) -> Dict[Tuple[str, str], float]:
    """
    Build a symmetric lookup (a, b) -> cosim from df5.
    All ids are cast to string to avoid type mismatches (e.g. int vs str).
    """
    lookup: Dict[Tuple[str, str], float] = {}

    sub = df5[[id1_col, id2_col, cosim_col]].copy()
    # Normalize ids as strings
    sub[id1_col] = sub[id1_col].astype(str)
    sub[id2_col] = sub[id2_col].astype(str)

    for a, b, s in sub.itertuples(index=False, name=None):
        lookup[(a, b)] = s
        lookup[(b, a)] = s  # enforce symmetry

    return lookup

def _ensure_list(x: Any) -> list:
    """
    Normalize a cell into a (possibly empty) list.

    - None / NaN -> []
    - list/tuple -> list(x)
    - anything else -> [x]
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def compute_neighbor_weighted_score(
    df5: pd.DataFrame,
    df6: pd.DataFrame,
    id1_col: str = "id1",
    id2_col: str = "id2",
    cosim_df5_col: str = "cosim",
    cosim_df6_col: str = "cosim",
    alpha_col: str = "alpha",
    eq1_col: str = "equivalents1",
    eq2_col: str = "equivalents2",
    new_col: str = "cos_sim_score",
) -> pd.DataFrame:
    """
    Compute neighbor-weighted aggregation scores for all rows in df6 using df5 as the similarity bank.
    
    This implements the "Free Entailment" aggregation logic:
    Score(A,B) = Sim(A,B) * [ alpha * Mean(Sim(B, Neighbors_A)) + (1-alpha) * Mean(Sim(A, Neighbors_B)) ]
    
    If 'cosim' columns contain Cosine Similarity, the output is the Cos Sim Score.
    If 'cosim' columns contain NLI Probabilities, the output is the NLI Free Score.

    Parameters
    ----------
    df5 : pd.DataFrame
        DataFrame with base similarities (id1, id2, sim, ...).
    df6 : pd.DataFrame
        DataFrame with (id1, id2, sim, alpha, equivalents1, equivalents2).
    ...
    new_col : str, default "cos_sim_score"
        Name of the column to create in df6.

    Returns
    -------
    pd.DataFrame
        Copy of df6 including an extra column `new_col` with the aggregated scores.
    """

    sigma_lookup = _build_sigma_lookup_from_df5(
        df5, id1_col=id1_col, id2_col=id2_col, cosim_col=cosim_df5_col
    )

    def get_sigma(a: Any, b: Any) -> float:
        """Symmetric lookup for σ_ab using df5. Casts key to str."""
        return sigma_lookup.get((str(a), str(b)), np.nan)

    def mean_sigma(anchor: Any, others: Iterable[Any]) -> float:
        """Mean similarity from `anchor` to all `others` using df5."""
        vals = [get_sigma(anchor, k) for k in others]
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            return 0.0
        return float(np.mean(vals))

    def row_score(row: pd.Series) -> float:
        i = row[id1_col]
        j = row[id2_col]
        alpha = row[alpha_col]
        sigma_ij = row[cosim_df6_col]

        eq_i = [k for k in _ensure_list(row[eq1_col]) if str(k) != str(i)]
        eq_j = [k for k in _ensure_list(row[eq2_col]) if str(k) != str(j)]
        
        if np.isnan(sigma_ij):
            return float(np.nan)

        # If both equivalence sets are empty, return raw similarity (Sim(A,B))
        if len(eq_i) == 0 and len(eq_j) == 0:
            return sigma_ij

        # If neighbors exist but alpha is NaN, we cannot perform weighting
        if np.isnan(alpha):
            return float(np.nan)

        term_i = alpha * mean_sigma(j, eq_i) if (alpha != 0 and len(eq_i) > 0) else 0.0
        term_j = (1.0 - alpha) * mean_sigma(i, eq_j) if (alpha != 1 and len(eq_j) > 0) else 0.0

        return sigma_ij * (term_i + term_j)

    df6_out = df6.copy()
    df6_out[new_col] = df6_out.apply(row_score, axis=1)
    return df6_out

def add_graph_features(
    df: pd.DataFrame,
    entailment_df: pd.DataFrame,
    id1_col: str,
    id2_col: str,
    verdict_col: str = "verdict",
    positive_label: str = "YES",
    decay: float = 0.9,
    max_hops: int = 5
) -> pd.DataFrame:
    """
    Computes advanced graph-based features:
    1. 'graph_entailment_score': A -> B reachability (decayed by path length).
    2. 'graph_equivalence_score': A <-> B bidirectional reachability (decayed).
    
    A direct link (1 hop) gets score 1.0.
    A 2-hop link gets score 1.0 * decay.
    """
    # 1. Build Directed Graph from known positive entailments
    print("Building Directed Entailment Graph...")
    graph = defaultdict(set)
    positives = entailment_df[entailment_df[verdict_col] == positive_label]
    
    for _, row in positives.iterrows():
        u = str(row[id1_col])
        v = str(row[id2_col])
        graph[u].add(v)
        
    # 2. Shortest Path BFS
    def get_shortest_path_len(start, end, graph, limit):
        if start == end: return 0
        if start not in graph: return None
        
        queue = [(start, 1)] # (node, depth) where depth is number of edges
        visited = {start}
        
        while queue:
            node, depth = queue.pop(0)
            if depth > limit:
                continue
                
            for neighbor in graph.get(node, []):
                if neighbor == end:
                    return depth
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return None

    fwd_scores = []
    bidir_scores = []
    
    print(f"Computing graph features for {len(df)} pairs...")
    
    for _, row in df.iterrows():
        u = str(row[id1_col])
        v = str(row[id2_col])
        
        # A -> B (Entailment)
        dist_ab = get_shortest_path_len(u, v, graph, max_hops)
        
        # B -> A (Reverse Entailment for Equivalence)
        dist_ba = get_shortest_path_len(v, u, graph, max_hops) if dist_ab is not None else None
        
        # Calculate Scores
        # 1. Forward (Entailment)
        if dist_ab is not None:
            # hop 1 = 1.0, hop 2 = decay, hop 3 = decay^2
            s_fwd = decay ** (dist_ab - 1)
        else:
            s_fwd = 0.0
            
        # 2. Bidirectional (Equivalence)
        # We take the geometric mean of the two path scores if both exist
        if dist_ab is not None and dist_ba is not None:
            s_ba = decay ** (dist_ba - 1)
            s_bidir = (s_fwd * s_ba) ** 0.5
        else:
            s_bidir = 0.0
             
        fwd_scores.append(s_fwd)
        bidir_scores.append(s_bidir)
        
    df['graph_entailment_score'] = fwd_scores
    df['graph_equivalence_score'] = bidir_scores
    
    return df


def add_graph_features_vectorized(
    df: pd.DataFrame,
    entailment_df: pd.DataFrame,
    id1_col: str = "id1",
    id2_col: str = "id2",
    verdict_col: str = "verdict",
    positive_label: str = "YES",
    decay: float = 0.9,
    max_hops: int = 5,
) -> pd.DataFrame:
    """
    Vectorized version of :func:`add_graph_features` suitable for millions of rows.

    Instead of running BFS per row (O(|df| * (V+E))), this:

    1. Builds the directed entailment graph.
    2. Runs *one* BFS from every unique node that has outgoing edges
       (limited to *max_hops*).  With a sparse graph this is extremely fast.
    3. Stores all reachable (source, target) → distance in a flat dict.
    4. Vectorised lookup via ``pd.Series.map()`` for all rows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with (id1, id2) pairs to score.
    entailment_df : pd.DataFrame
        Known entailments with a *verdict_col* column.
    id1_col, id2_col : str
        Column names for the pair IDs.
    verdict_col : str
        Column that holds the entailment label.
    positive_label : str
        Value in *verdict_col* indicating positive entailment.
    decay : float
        Decay factor applied per hop (score = decay^(hops-1)).
    max_hops : int
        Maximum BFS depth.

    Returns
    -------
    pd.DataFrame
        *df* with ``graph_entailment_score`` and ``graph_equivalence_score``
        columns added in-place.
    """
    from collections import deque

    # 1. Build directed graph
    graph: Dict[str, set] = defaultdict(set)
    positives = entailment_df[entailment_df[verdict_col] == positive_label]
    for row in positives.itertuples(index=False):
        u = str(getattr(row, id1_col))
        v = str(getattr(row, id2_col))
        graph[u].add(v)

    print(f"  Graph: {len(graph)} source nodes, "
          f"{sum(len(v) for v in graph.values())} edges")

    # 2. BFS from every source node (only nodes with outgoing edges matter)
    #    Returns {target_node: hop_distance} for reachable targets within max_hops.
    def _bfs(start: str) -> Dict[str, int]:
        dists: Dict[str, int] = {}
        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            node, depth = queue.popleft()
            if depth > 0:
                dists[node] = depth
            if depth >= max_hops:
                continue
            for neighbour in graph.get(node, ()):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((neighbour, depth + 1))
        return dists

    # Collect all unique nodes that appear in df AND have outgoing edges in graph
    ids1 = df[id1_col].astype(str)
    ids2 = df[id2_col].astype(str)
    nodes_to_bfs = (set(ids1) | set(ids2)) & set(graph.keys())

    print(f"  BFS from {len(nodes_to_bfs)} nodes (max_hops={max_hops})...")

    # Flat lookup:  "source|target" → hop distance
    pair_dist: Dict[str, int] = {}
    for src in nodes_to_bfs:
        for tgt, d in _bfs(src).items():
            pair_dist[f"{src}|{tgt}"] = d

    print(f"  Reachable pairs: {len(pair_dist):,}")

    # 3. Vectorised lookup
    keys_ab = ids1 + '|' + ids2
    keys_ba = ids2 + '|' + ids1

    dist_ab = keys_ab.map(pair_dist)
    dist_ba = keys_ba.map(pair_dist)

    # 4. Compute scores
    fwd = np.where(dist_ab.notna(), decay ** (dist_ab.fillna(0) - 1), 0.0)
    bwd = np.where(dist_ba.notna(), decay ** (dist_ba.fillna(0) - 1), 0.0)
    bidir = np.where(
        dist_ab.notna() & dist_ba.notna(),
        np.sqrt(fwd * bwd),
        0.0,
    )

    df['graph_entailment_score'] = fwd.astype(np.float32)
    df['graph_equivalence_score'] = bidir.astype(np.float32)

    n_nonzero_ent = (df['graph_entailment_score'] > 0).sum()
    n_nonzero_eq  = (df['graph_equivalence_score'] > 0).sum()
    print(f"  ✓ graph_entailment_score: {n_nonzero_ent:,} non-zero / {len(df):,}")
    print(f"  ✓ graph_equivalence_score: {n_nonzero_eq:,} non-zero / {len(df):,}")

    return df


### PREDICT ENTAILMENT
def train_entailment_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    method: Literal["logistic", "spline", "kernel", "tree", "probit", "boosting"] = "logistic",
    positive_label: str = "YES",
    **kwargs
):
    """
    Trains a model to predict entailment based on score features.
    Returns the trained pipeline (Scaler + Model).
    
    Methods:
    - 'logistic': Standard Logistic Regression
    - 'spline':   Logistic Regression with Spline features (Non-linear)
    - 'kernel':   Support Vector Classifier (RBF Kernel) with probability=True
    - 'tree':     Decision Tree Classifier
    - 'boosting': Histogram Gradient Boosting Classifier
    - 'probit':   (Not implemented in sklearn version, falls back to Logistic with warning)
    """
    # Filter valid data
    df_clean = df.dropna(subset=feature_cols + [target_col])
    
    X = df_clean[feature_cols].values
    # Convert target to 0/1
    y = (df_clean[target_col] == positive_label).astype(int).values
    
    steps = [StandardScaler()]
    
    if method == "logistic":
        model = LogisticRegression(class_weight='balanced', solver='liblinear')
        print("Training Logistic Regression...")
        
    elif method == "spline":
        # Spline Regression in classification context:
        # Transform features -> Splines, then apply Logistic Regression
        steps.append(SplineTransformer(n_knots=kwargs.get("n_knots", 5), degree=kwargs.get("degree", 3)))
        model = LogisticRegression(class_weight='balanced', solver='liblinear')
        print("Training Spline Logistic Regression...")
        
    elif method == "kernel":
        # Kernel Method: Kernel SVM (RBF)
        # Note: probability=True uses Platts scaling (internal cross-validation), simpler but slower
        model = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
        print("Training Kernel SVM (RBF)...")

    elif method == "tree":
        # Decision Tree Classifier
        # Uses max_depth to prevent overfitting (default 5 if not provided in kwargs)
        model = DecisionTreeClassifier(
            class_weight='balanced', 
            random_state=42, 
            max_depth=kwargs.get("max_depth", 5)
        )
        print("Training Decision Tree Classifier...")
        
    elif method == "boosting":
        # Enhancements for Boosting:
        # 1. Balanced class weights (critical for low-send detection)
        # 2. Monotonic constraints (optional): Force positive correlation for similarity metrics
        
        # Check if user wants monotonic constraints
        monotonic_cst = kwargs.get("monotonic_cst", None)
        if monotonic_cst is None and kwargs.get("enforce_monotonicity", False):
            # Assuming all features are similarity scores (positive correlation) -> 1
            # If feature is something like "distance", it should be -1.
            # Here we assume standard free entailment features are positive.
            monotonic_cst = [1] * len(feature_cols)
            print(f"Enforcing monotonic constraints: {monotonic_cst}")

        model = HistGradientBoostingClassifier(
            random_state=42,
            class_weight='balanced',
            learning_rate=kwargs.get("learning_rate", 0.05), # Slightly lower conservative LR
            max_iter=kwargs.get("max_iter", 200),
            l2_regularization=kwargs.get("l2_regularization", 1.0),
            monotonic_cst=monotonic_cst
        )
        print(f"Training Histogram Gradient Boosting Classifier (lr={model.learning_rate}, iter={model.max_iter})...")

    else:
        raise ValueError(f"Unknown method '{method}'")

    steps.append(model)
    pipeline = make_pipeline(*steps)
    
    pipeline.fit(X, y)
    
    # Optional: Print rough accuracy
    acc = pipeline.score(X, y)
    print(f"Model ({method}) Train Accuracy: {acc:.4f}")
    
    return pipeline

def predict_entailment_probabilities(
    df: pd.DataFrame,
    model_pipeline,
    feature_cols: List[str],
    new_col: str = "entailment_prob",
    transitivity_col: str = "transitivity_score",
    chunk_size: int = 2_000_000,
) -> pd.DataFrame:
    """
    Uses the trained regression model to predict probability (0 to 1).
    Modified: works IN-PLACE (no df.copy()) and predicts in chunks to
    avoid allocating a 75 M-row feature matrix at once.
    """
    n = len(df)
    probs_all = np.full(n, np.nan, dtype=np.float64)

    mask_valid = df[feature_cols].notna().all(axis=1).values

    valid_idx = np.where(mask_valid)[0]
    if len(valid_idx) > 0:
        # Predict in chunks to limit peak RAM
        n_pred_chunks = (len(valid_idx) + chunk_size - 1) // chunk_size
        for ci in range(n_pred_chunks):
            clo = ci * chunk_size
            chi = min(clo + chunk_size, len(valid_idx))
            idx_slice = valid_idx[clo:chi]
            X_chunk = df.iloc[idx_slice][feature_cols].values
            probs_chunk = model_pipeline.predict_proba(X_chunk)[:, 1]
            probs_all[idx_slice] = probs_chunk
            del X_chunk, probs_chunk

    df[new_col] = probs_all
    return df

def compare_entailment_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_names: List[str] = ["logistic", "spline", "tree", "boosting"],
    positive_label: str = "YES",
    **kwargs
) -> Tuple[pd.DataFrame, str]:
    """
    Trains multiple models and compares them using ROC-AUC, Log Loss, and Separation.
    Returns a DataFrame of results and the name of the best model (highest ROC-AUC).
    Accepts **kwargs to pass hyperparameters (like enforce_monotonicity=True).
    """
    results_data = []

    # Filter valid data once
    df_clean = df.dropna(subset=feature_cols + [target_col])

    if df_clean.empty:
        raise ValueError("No valid data found for training/comparison.")

    print(f"Running comparative analysis on {len(df_clean)} samples...")
    print(f"Features: {feature_cols}\n")

    for m_name in model_names:
        print(f"--- Training {m_name} ---")
        try:
            # 1. Train using the existing helper
            pipeline = train_entailment_model(
                df=df_clean,
                feature_cols=feature_cols,
                target_col=target_col,
                method=m_name,
                positive_label=positive_label,
                **kwargs
            )

            # 2. Predict Probabilities using Cross Validation (Simulating Test Data)
            # This prevents overestimating the performance of powerful models like Boosting
            X = df_clean[feature_cols].values
            y_true = (df_clean[target_col] == positive_label).astype(int).values
            
            # Dynamically set CV folds based on minority class size
            min_class_count = int(np.bincount(y_true).min()) if len(y_true) > 0 else 0
            cv_folds = min(5, min_class_count)
            
            if cv_folds >= 2:
                # We use cross_val_predict to generate "clean" predictions for every row
                # The model is trained on K-1 folds and predicts on the Kth fold.
                y_probs = cross_val_predict(
                    pipeline, 
                    X, 
                    y_true, 
                    cv=cv_folds, 
                    method='predict_proba'
                )[:, 1]
            else:
                # Too few samples for CV — use training predictions as fallback
                pipeline.fit(X, y_true)
                y_probs = pipeline.predict_proba(X)[:, 1]
                print(f"  ⚠ Too few samples for CV ({len(y_true)} total, min class={min_class_count}). Using train predictions.")

            # 3. Calculate Metrics on these "Out-of-Sample" predictions
            roc_auc = roc_auc_score(y_true, y_probs)
            ll = log_loss(y_true, y_probs)

            # Separation
            yes_probs = y_probs[y_true == 1]
            no_probs = y_probs[y_true == 0]
            separation = yes_probs.mean() - no_probs.mean() if (len(yes_probs) > 0 and len(no_probs) > 0) else np.nan

            results_data.append({
                "Model": m_name,
                "ROC-AUC (CV)": roc_auc, # Renamed to clarify it's CV score
                "Log Loss": ll,
                "Separation": separation,
                "Mean Prob (YES)": yes_probs.mean() if len(yes_probs) > 0 else np.nan,
                "Mean Prob (NO)": no_probs.mean() if len(no_probs) > 0 else np.nan,
            })

        except Exception as e:
            print(f"Failed to train {m_name}: {e}")

    summary_df = pd.DataFrame(results_data).set_index("Model")
    # Sort by the new column name
    summary_df = summary_df.sort_values("ROC-AUC (CV)", ascending=False)

    if not summary_df.empty:
        best_model = summary_df.index[0]
    else:
        best_model = None

    return summary_df, best_model


def optimize_boosting_hyperparameters(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    positive_label: str = "YES",
    n_trials: int = 50,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Uses Optuna to find the best hyperparameters for HistGradientBoostingClassifier
    maximizing ROC-AUC.
    """
    try:
        import optuna
    except ImportError:
        print("Optuna not installed. Please run: pip install optuna")
        return {}
    
    # Filter valid data
    df_clean = df.dropna(subset=feature_cols + [target_col])
    if df_clean.empty:
        raise ValueError("No valid data for optimization.")
        
    X = df_clean[feature_cols].values
    y = (df_clean[target_col] == positive_label).astype(int).values
    
    # Suppress Optuna logging to avoid clutter
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        # Hyperparameters space
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 63),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True),
            'random_state': random_state,
            'class_weight': 'balanced'
        }
        
        model = HistGradientBoostingClassifier(**params)
        
        # Dynamically set CV folds based on minority class size
        min_class_count = int(np.bincount(y).min()) if len(y) > 0 else 0
        cv_folds = min(3, min_class_count)
        
        if cv_folds >= 2:
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='roc_auc')
            return scores.mean()
        else:
            # Too few samples for CV — fit and score on training data
            model.fit(X, y)
            y_probs = model.predict_proba(X)[:, 1]
            return roc_auc_score(y, y_probs)

    print(f"Starting Optuna optimization with {n_trials} trials...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best ROC-AUC: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    
    return study.best_params


#THRESHOLDS


def find_best_thresholds(
    df: pd.DataFrame,
    score_col: str = "cos_sim_score",
    verdict_col: str = "verdict",
    positive_label: str = "YES",
) -> Dict[str, Any]:
    """
    Given a dataframe with columns:
        - score_col (e.g. 'cos_sim_score')
        - verdict_col (values 'YES' / 'NO')

    find thresholds tau (floats) for the decision rule

        y_hat = 1 if score > tau else 0
        
    Note: "Sending to LLM" in this context refers to candidates with score > tau.
    (i.e. we use the model to filter out low-scoring pairs as 'No', and send high-scoring to LLM for verification)

    that respectively maximize:
        - accuracy
        - F1
        - number of true positives (TP)
        - precision
        - recall

    and return:
        - the best tau for each metric,
        - the full per-threshold metrics table,
        - a compact table with rows only for the best taus.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least score_col and verdict_col.
    score_col : str, default "cos_sim_score"
        Name of the column with the scores.
    verdict_col : str, default "verdict"
        Name of the column with the ground-truth labels ('YES'/'NO').
    positive_label : str, default "YES"
        Value in verdict_col that is considered as positive (1).

    Returns
    -------
    dict
        Dictionary with keys:
            - 'best_tau_accuracy', 'best_accuracy'
            - 'best_tau_f1',       'best_f1'
            - 'best_tau_tp',       'max_true_positives'
            - 'best_tau_precision','best_precision'
            - 'best_tau_recall',   'best_recall'
            - 'thresholds_df'      (full table)
            - 'best_taus_table'    (only rows for best taus)
    """

    # Keep only rows with non-missing score and verdict
    mask_valid = df[score_col].notna() & df[verdict_col].notna()
    df_valid = df.loc[mask_valid, [score_col, verdict_col]].copy()

    if df_valid.empty:
        raise ValueError("No valid rows with both score and verdict.")

    # Encode verdict as 0/1
    y_true = (df_valid[verdict_col] == positive_label).astype(int).to_numpy()
    scores = df_valid[score_col].astype(float).to_numpy()

    # Unique candidate thresholds (we will use '>' rule)
    unique_scores = np.unique(scores)

    records = []
    N = len(y_true)

    for tau in unique_scores:
        y_pred = (scores > tau).astype(int)

        TP = int(((y_true == 1) & (y_pred == 1)).sum())
        TN = int(((y_true == 0) & (y_pred == 0)).sum())
        FP = int(((y_true == 0) & (y_pred == 1)).sum())
        FN = int(((y_true == 1) & (y_pred == 0)).sum())

        acc = (TP + TN) / N

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        records.append(
            {
                "tau": float(tau),
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    thresholds_df = pd.DataFrame.from_records(records)

    # Pick best thresholds (ties broken by first occurrence, i.e. smallest tau)
    idx_best_acc  = thresholds_df["accuracy"].idxmax()
    idx_best_f1   = thresholds_df["f1"].idxmax()
    idx_best_tp   = thresholds_df["TP"].idxmax()
    idx_best_prec = thresholds_df["precision"].idxmax()
    idx_best_rec  = thresholds_df["recall"].idxmax()

    best_acc_row  = thresholds_df.loc[idx_best_acc]
    best_f1_row   = thresholds_df.loc[idx_best_f1]
    best_tp_row   = thresholds_df.loc[idx_best_tp]
    best_prec_row = thresholds_df.loc[idx_best_prec]
    best_rec_row  = thresholds_df.loc[idx_best_rec]

    # ---------------------------------------------------------
    # New: Find thresholds for 1-5% Send Rate
    # "Send to LLM" means Predicted Positive/Candidate (Score > tau)
    # We want tau s.t. P(Score > tau) in [0.01, 0.05]
    # We look at high percentiles (95th-99th)
    # ---------------------------------------------------------
    low_send_metrics = []
    # Target percentiles for keeping data (top 1-5%)
    # e.g. 0.99 quantile -> 1% of data is above it.
    target_percentiles = [0.99, 0.98, 0.97, 0.96, 0.95]
    
    for p in target_percentiles:
        try:
            t = float(np.quantile(scores, p))
        except:
            t = 0.0
        
        # Calculate stats for this specific t
        y_pred = (scores > t).astype(int)
        
        # TP = Sent & Actually YES
        TP_ls = int(((y_true == 1) & (y_pred == 1)).sum())
        # FN = Auto-Refused & Actually YES (Loss)
        FN_ls = int(((y_true == 1) & (y_pred == 0)).sum())
        # FP = Sent & Actually NO (Wasted Cost)
        FP_ls = int(((y_true == 0) & (y_pred == 1)).sum())
        # TN = Auto-Refused & Actually NO (Efficiency)
        TN_ls = int(((y_true == 0) & (y_pred == 0)).sum())

        # Sent count is number of 1 predictions (Predicted Positive / Candidate)
        sent_count = int((y_pred == 1).sum())
        sent_rate = sent_count / N if N > 0 else 0
        
        # We want to minimize FN (Valid candidates that we Auto-Refused)
        # Or maximize Precision? 
        # Usually for low send rate, we want to make sure the ones we send are worth it?
        # Or make sure we didn't lose too many? 
        # Let's track FN as the "error" of the refusal.
        
        low_send_metrics.append({
            "target_percentile": 1 - p,
            "tau": t,
            "sent_rate": sent_rate,
            "FN": FN_ls,
            "TP": TP_ls,
            "FP": FP_ls,
            "TN": TN_ls
        })
        
    low_send_df = pd.DataFrame(low_send_metrics)
    
    # Select best: Minimize FN among these?
    # Actually if we send equally small amounts, the one with lowest FN (highest Recall) is best.
    if not low_send_df.empty:
        # Sort by FN ascending, break ties with sent_rate
        best_low_send_row = low_send_df.sort_values("FN", ascending=True).iloc[0]
        best_tau_low_send = float(best_low_send_row["tau"])
    else:
        best_tau_low_send = 0.0

    # Collect unique best taus
    best_taus = {
        float(best_acc_row["tau"]),
        float(best_f1_row["tau"]),
        float(best_tp_row["tau"]),
        float(best_prec_row["tau"]),
        float(best_rec_row["tau"])
    }

    # Small table with only rows for best taus
    best_taus_table = (
        thresholds_df[thresholds_df["tau"].isin(best_taus)]
        .sort_values("tau")
        .reset_index(drop=True)
    )

    return {
        "best_tau_accuracy":  float(best_acc_row["tau"]),
        "best_accuracy":      float(best_acc_row["accuracy"]),

        "best_tau_f1":        float(best_f1_row["tau"]),
        "best_f1":            float(best_f1_row["f1"]),

        "best_tau_tp":        float(best_tp_row["tau"]),
        "max_true_positives": int(best_tp_row["TP"]),

        "best_tau_precision": float(best_prec_row["tau"]),
        "best_precision":     float(best_prec_row["precision"]),

        "best_tau_recall":    float(best_rec_row["tau"]),
        "best_recall":        float(best_rec_row["recall"]),
        
        "best_tau_low_send":  best_tau_low_send,
        "low_send_table":     low_send_df,

        "thresholds_df":      thresholds_df,
        "best_taus_table":    best_taus_table,
    }


def plot_llm_savings_over_thresholds(
    df: pd.DataFrame,
    prob_col: str,
    verdict_col: str,
    positive_label: str = "YES",
    step: float = 0.001,
    markers: Dict[str, float] = None,
    show_optimal_lines: bool = True,
    cost_fn: float = 5.0,
    beta_fp: float = 0.5
):
    """
    Plots the 'Sent to LLM' rate vs Threshold using Plotly.
    Here, 'Sent to LLM' means Score > Threshold.
    RETURNS the figure object instead of just showing it.
    """
    thresholds = np.arange(0, 1.0 + step, step)
    results = []
    
    # Pre-calculate boolean series for efficiency
    actual_pos = (df[verdict_col] == positive_label).values
    actual_neg = ~actual_pos
    probs = df[prob_col].values
    total = len(df)
    
    # Calculate optimal thresholds if requested
    if markers is None:
        markers = {}
        
    if show_optimal_lines:
        y_true = (df[verdict_col] == positive_label).astype(int).values
        y_probs = df[prob_col].values
        
        # Min FP (Optimize Precision)
        try:
            thresh_fp, f_beta = get_optimal_threshold_minimize_fp(y_true, y_probs, beta=beta_fp)
            markers[f"Min FP (Beta={beta_fp})"] = thresh_fp
        except Exception as e:
            print(f"Could not calculate Min FP threshold: {e}")

        # Min FN (Cost Sensitive)
        try:
            # Using Cost strategy (default)
            thresh_fn = get_optimal_threshold_minimize_fn(strategy='cost', cost_fn=cost_fn)
            markers[f"Min FN (Cost FN={cost_fn})"] = thresh_fn
        except Exception as e:
            print(f"Could not calculate Min FN threshold: {e}")
            

    for t in thresholds:
        # Decision: 
        # Sent if prob > t
        # Auto-Reject if prob <= t
        send_mask = (probs > t)
        reject_mask = ~send_mask
        
        # Confusion components
        # Sent_Yes = Send & Pos
        # Sent_No = Send & Neg (Over-sending)
        sent_pos = np.int64(np.sum(send_mask & actual_pos))
        sent_neg = np.int64(np.sum(send_mask & actual_neg))
        
        # Reject_No = Reject & Neg (Efficiency)
        # Reject_Yes = Reject & Pos (Lost Opportunity / FN)
        tn_rej = np.int64(np.sum(reject_mask & actual_neg))
        fn_rej = np.int64(np.sum(reject_mask & actual_pos))
        
        sent_count = sent_pos + sent_neg
        sent_rate = sent_count / total
        
        results.append({
            "threshold": t,
            "sent_rate": sent_rate,
            "Sent_Yes": sent_pos,
            "Sent_No": sent_neg,
            "FN_Reject": fn_rej,
            "TN_Reject": tn_rej
        })
        
    df_plot = pd.DataFrame(results)
    
    fig = go.Figure()
    
    # Main Curve
    fig.add_trace(go.Scatter(
        x=df_plot['threshold'], 
        y=df_plot['sent_rate'],
        mode='lines+markers',
        name='Sent to LLM %',
        marker=dict(size=4, color='royalblue'),
        hovertemplate=(
            "<b>Threshold: %{x:.2f}</b><br>" +
            "Sent to LLM: %{y:.1%}<br>" +
            "Sent_Yes (Verified Pos): %{customdata[0]}<br>" +
            "Sent_No (Verified Neg): %{customdata[1]}<br>" +
            "FN (Lost Opportunity): %{customdata[2]}<br>" +
            "TN (Efficiency): %{customdata[3]}<extra></extra>"
        ),
        customdata=df_plot[['Sent_Yes', 'Sent_No', 'FN_Reject', 'TN_Reject']].values
    ))
    
    # Add Markers (Vertical Lines)
    if markers:
        colors = ['red', 'green', 'purple', 'orange', 'cyan']
        i = 0
        for name, thresh in markers.items():
            if thresh is None or np.isnan(thresh):
                continue
                
            color = colors[i % len(colors)]
            i += 1
            
            fig.add_vline(
                x=thresh, 
                line_width=2, 
                line_dash="dot", 
                line_color=color, 
                annotation_text=name, 
                annotation_position="top right"
            )
            
            # Optional: Add a point on the curve
            closest_idx = (np.abs(df_plot['threshold'] - thresh)).argmin()
            row = df_plot.iloc[closest_idx]
            
            fig.add_trace(go.Scatter(
                x=[row['threshold']],
                y=[row['sent_rate']],
                mode='markers',
                marker=dict(color=color, size=10, symbol='star'),
                name=shorten_name(name),
                showlegend=False,
                hoverinfo='skip'
            ))

    fig.update_layout(
        title="LLM Workload vs. tau Threshold (Sending > tau)",
        xaxis_title="Threshold (Probability Cutoff)",
        yaxis_title="Percentage Sent to LLM",
        yaxis_tickformat='.0%',
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig  # Return the object instead of showing

def shorten_name(name, max_len=15):
    if len(name) <= max_len: return name
    return name[:12] + "..."


def get_optimal_threshold_minimize_fp(y_true, y_probs, beta=0.5):
    """
    Finds the optimal threshold to minimize False Positives (optimize Precision).
    Uses the F-beta score with beta < 1 (default 0.5).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    # Avoid zero division
    numerator = (1 + beta**2) * (precisions * recalls)
    denominator = (beta**2 * precisions) + recalls
    fbeta_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    # Locate the best threshold
    ix = np.argmax(fbeta_scores)
    
    if ix < len(thresholds):
        best_thresh = thresholds[ix]
    else:
        best_thresh = thresholds[-1] 
        
    return best_thresh, fbeta_scores[ix]


def get_optimal_threshold_minimize_fn(y_true=None, y_probs=None, cost_fp=1, cost_fn=5, strategy='cost'):
    """
    Finds the optimal threshold to minimize False Negatives (optimize Recall).
    Two strategies:
    1. 'cost' (Default): Uses Cost-Sensitive Bayes Risk: tau = C_fp / (C_fp + C_fn).
       Does not require y_true/y_probs.
    2. 'f2': Maximizes F2 score (beta=2). Requires y_true and y_probs.
    """
    if strategy == 'cost':
        return cost_fp / (cost_fp + cost_fn)
    
    if strategy == 'f2':
        if y_true is None or y_probs is None:
            raise ValueError("Strategy 'f2' requires y_true and y_probs.")
        
        # We reuse logic similar to minimizing FP but with beta=2
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        beta = 2.0
        numerator = (1 + beta**2) * (precisions * recalls)
        denominator = (beta**2 * precisions) + recalls
        f_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        
        ix = np.argmax(f_scores)
        if ix < len(thresholds):
            return thresholds[ix]
        else:
            return thresholds[-1]
            
    raise ValueError(f"Unknown strategy: {strategy}")


def estimate_deepseek_cost(
    df: pd.DataFrame, 
    prob_col: str, 
    threshold: float, 
    avg_text_char_len: int = 200, 
    model: str = "deepseek-reasoner",
):
    """
    Estimates the cost of running entailment pairs through DeepSeek V3/R1.
    Calculates cost ONLY for pairs that would be sent to the LLM (Prob > Threshold).
    
    Assumptions based on user prompt structure:
    - System Prompt: "You are an expert logician..." (~50 tokens)
    - User Prompt Template: "A->B? B->A? Contradiction? Neutral? Paraphrase?" (~120 tokens)
    - Input Texts: 2 texts per pair.
    - Output: Reasoning trace + 5 classification answers.
    
    Calibration (from User Data):
    - 448,370 pairs cost ~$850 using the reasoner model.
    - This implies ~1.90 USD per 1,000 pairs.
    - Implied average output tokens (CoT + Answer) is approx 800 tokens.
    
    Pricing (Approximate, check current API pricing):
    - DeepSeek-V3 (Standard): Input $0.14/1M, Output $0.28/1M
    - DeepSeek-R1 (Reasoner): Input $0.55/1M, Output $2.19/1M (Includes CoT tokens)
    """
    # Filter pairs that will be sent (Prob > Threshold means "Send to LLM")
    n_pairs = int((df[prob_col] > threshold).sum())
    
    if n_pairs == 0:
        print(f"--- Cost Estimation (Threshold {threshold:.4f}) ---")
        print("0 Pairs selected (None > Threshold). Cost: $0.00")
        return 0.0

    # --- Token Estimation ---
    # 1 token ~= 4 chars typically, but logical/legal text might be denser. 
    # Let's use a conservative 3 chars/token for safe estimation.
    avg_text_tokens = (avg_text_char_len * 2) / 3.0  # Two texts (A and B)
    
    system_prompt_tokens = 50
    user_template_tokens = 120
    
    input_tokens_per_pair = system_prompt_tokens + user_template_tokens + avg_text_tokens
    
    # Output estimation
    if "reasoner" in model.lower() or "r1" in model.lower():
        # Calibrated based on 448k pairs costing $850
        output_tokens_per_pair = 800  
        # Pricing: $0.55 / $2.19 per 1M
        price_in_per_1m = 0.55
        price_out_per_1m = 2.19
    else:
        # Standard V3
        output_tokens_per_pair = 200
        # Pricing: $0.14 / $0.28 per 1M
        price_in_per_1m = 0.14
        price_out_per_1m = 0.28
        
    total_input_tokens = n_pairs * input_tokens_per_pair
    total_output_tokens = n_pairs * output_tokens_per_pair
    
    cost_input = (total_input_tokens / 1_000_000) * price_in_per_1m
    cost_output = (total_output_tokens / 1_000_000) * price_out_per_1m
    total_cost = cost_input + cost_output
    
    print(f"--- Cost Estimation for {n_pairs:,} Pairs (P > {threshold:.4f}) ---")
    print(f"Model: {model}")
    print(f"Input Tokens:  {total_input_tokens:,.0f} (${cost_input:.4f})")
    print(f"Output Tokens: {total_output_tokens:,.0f} (${cost_output:.4f})")
    print(f"Total Cost:    ${total_cost:.4f} (Approx ${total_cost/n_pairs*1000:.2f}/1k pairs)")
    
    return total_cost


def generate_final_df(
    df: pd.DataFrame, 
    prob_col: str, 
    threshold: float,
    keep_cols: List[str] = ['id1', 'id2', 'text1', 'text2', 'entailment_probability'],
    df_clause: Optional[pd.DataFrame] = None,
    id_col: str = 'sentence_id',
    text_col: str = 'sentence',
):
    """
    Filters the dataframe to return only the pairs that need to be sent to the LLM.
    Criteria: Entailment Probability > Threshold.
    (i.e. Low scores are Auto-Rejected, High scores are Sent for Verification).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    prob_col : str
        Column containing the computed probabilities.
    threshold : float
        Cutoff threshold.
    keep_cols : list
        List of columns to keep in the final output. 
        If 'entailment_probability' is in this list but not in df, 
        prob_col will be renamed to 'entailment_probability'.
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with selected columns.
    """
    mask = df[prob_col] > threshold
    n_total = len(df)
    
    # FILTER FIRST — on 75M rows, only ~1k pass the threshold, so we
    # restrict all subsequent work to the tiny filtered subset.
    df_filtered = df.loc[mask].copy()

    # Handle renaming only if prob_col is different but 'entailment_probability' is requested
    if prob_col != 'entailment_probability' and 'entailment_probability' in keep_cols:
        df_filtered = df_filtered.rename(columns={prob_col: 'entailment_probability'})

    # Lazy text lookup: add text1/text2 from df_clause if missing
    if df_clause is not None:
        _id_to_text = dict(zip(df_clause[id_col].astype(str), df_clause[text_col]))
        if 'text1' not in df_filtered.columns and 'text1' in keep_cols:
            df_filtered['text1'] = df_filtered['id1'].astype(str).map(_id_to_text)
        if 'text2' not in df_filtered.columns and 'text2' in keep_cols:
            df_filtered['text2'] = df_filtered['id2'].astype(str).map(_id_to_text)

    # Ensure columns exist before selecting
    missing = [c for c in keep_cols if c not in df_filtered.columns]
    if missing:
        raise ValueError(f"Missing required columns for final output: {missing}. "
                         f"Pass df_clause= to auto-add text columns.")
        
    df_out = df_filtered[keep_cols]
    
    print("--- Generating LLM Batch ---")
    print(f"Original Count: {n_total:,}")
    pct = f"{(len(df_out)/n_total):.1%}" if n_total > 0 else "N/A"
    print(f"Filtered Count: {len(df_out):,} ({pct})")
    print(f"Condition:      P > {threshold:.4f} (Send High Confidence Pairs)")
    
    return df_out


def add_verdict(
    df: pd.DataFrame,
    id1_col: str = 'sentence_id_1',
    id2_col: str = 'sentence_id_2', 
    conclusion_col: str = 'llm_conclusion_12',
    positive_label: str = 'YES',
    new_col: str = 'verdict'
) -> pd.DataFrame:
    """
    Add a 'verdict' column that is YES only if pairs are entailed in BOTH directions.
    
    This function expects the dataframe to contain pairs in both directions:
    - (A, B) with llm_conclusion_12 indicating if A entails B
    - (B, A) with llm_conclusion_12 indicating if B entails A
    
    The verdict is YES only if both directions show entailment.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing pairwise LLM results with both directions.
    id1_col : str, default 'sentence_id_1'
        Column name for first sentence ID.
    id2_col : str, default 'sentence_id_2'
        Column name for second sentence ID.
    conclusion_col : str, default 'llm_conclusion_12'
        Column name containing the entailment conclusion (YES/NO).
    positive_label : str, default 'YES'
        The label indicating entailment in the conclusion column.
    new_col : str, default 'verdict'
        Name of the new verdict column to add.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added verdict column.
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'sentence_id_1': ['A', 'B', 'C', 'D'],
    ...     'sentence_id_2': ['B', 'A', 'D', 'C'],
    ...     'llm_conclusion_12': ['YES', 'YES', 'YES', 'NO']
    ... })
    >>> df_with_verdict = add_verdict(df)
    >>> # Pairs (A,B) and (B,A) both YES -> verdict YES
    >>> # Pairs (C,D) YES but (D,C) NO -> verdict NO
    """
    df_out = df.copy()
    
    # Create a sorted pair identifier for grouping
    def make_pair_key(row):
        id1, id2 = row[id1_col], row[id2_col]
        return tuple(sorted([str(id1), str(id2)]))
    
    df_out['_pair_key'] = df_out.apply(make_pair_key, axis=1)
    
    # For each pair, check if both directions have positive conclusion
    pair_verdicts = {}
    
    for pair_key, group in df_out.groupby('_pair_key'):
        # Check if we have both directions
        conclusions = group[conclusion_col].str.strip().str.upper()
        
        # Verdict is YES only if ALL rows for this pair show entailment
        # (should be 2 rows if both directions exist, or 1 if only one direction)
        if len(group) >= 2:
            # Both directions present - need both to be YES
            verdict = positive_label if all(conclusions == positive_label.upper()) else 'NO'
        else:
            # Only one direction - verdict matches that single conclusion
            verdict = positive_label if conclusions.iloc[0] == positive_label.upper() else 'NO'
        
        pair_verdicts[pair_key] = verdict
    
    # Map verdicts back to dataframe
    df_out[new_col] = df_out['_pair_key'].map(pair_verdicts)
    
    # Clean up temporary column
    df_out = df_out.drop(columns=['_pair_key'])
    
    # Report statistics
    n_yes = (df_out[new_col] == positive_label).sum()
    n_total = len(df_out)
    print(f"\n{'='*70}")
    print(f"VERDICT SUMMARY")
    print(f"{'='*70}")
    print(f"Total pairs: {n_total}")
    if n_total > 0:
        print(f"Bidirectional entailment (YES): {n_yes} ({n_yes/n_total:.1%})")
        print(f"Not bidirectionally entailed (NO): {n_total - n_yes} ({(n_total-n_yes)/n_total:.1%})")
    else:
        print(f"Bidirectional entailment (YES): 0")
        print(f"Not bidirectionally entailed (NO): 0")
    print(f"{'='*70}\n")
    
    return df_out


def calculate_entailment_ratio(
    df: pd.DataFrame,
    original_llm_file: str = "labeled_pairs/Results_DS_BtoS_iteration_0.csv",
    id1_col: str = 'sentence_id_1',
    id2_col: str = 'sentence_id_2',
    conclusion_col: str = 'llm_conclusion_12',
    positive_label: str = 'YES'
) -> dict:
    """
    Calculate the ratio of entailed pairs in df that are actually bidirectionally entailed 
    in the original LLM data.
    
    This function uses add_verdict() to determine which pairs are bidirectionally entailed 
    (YES in both directions) in the ground truth, then checks how many pairs selected by 
    FEA match those bidirectionally entailed pairs.
    
    Parameters
    ----------
    df : pd.DataFrame
        Output DataFrame from FEA_Pipeline containing pairs to be sent to LLM.
        Must have columns id1_col and id2_col.
    original_llm_file : str, default="Results_DS_BtoS.csv"
        Path to the original LLM labeled data file.
    id1_col : str, default='sentence_id_1'
        Column name for first sentence ID.
    id2_col : str, default='sentence_id_2'
        Column name for second sentence ID.
    conclusion_col : str, default='llm_conclusion_12'
        Column name containing LLM entailment conclusion in original file.
    positive_label : str, default='YES'
        Value indicating entailment in the conclusion column.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'total_pairs': Total number of pairs in df
        - 'bidirectionally_entailed': Number of pairs bidirectionally entailed in ground truth
        - 'entailment_ratio': Ratio of bidirectionally entailed pairs (entailed/total)
    
    Examples
    --------
    >>> df_to_llm = pd.DataFrame({
    ...     'sentence_id_1': ['A', 'B', 'C'],
    ...     'sentence_id_2': ['D', 'E', 'F'],
    ...     'score': [0.8, 0.7, 0.6]
    ... })
    >>> stats = calculate_entailment_ratio(df_to_llm)
    >>> print(f"Entailment ratio: {stats['entailment_ratio']:.1%}")
    """
    # Load original LLM data and get bidirectional verdicts.
    # If 'verdict' column already exists (from process_llm_results_bidirectional),
    # use it directly instead of recomputing with add_verdict.
    df_original = pd.read_csv(original_llm_file)
    if 'verdict' in df_original.columns and df_original['verdict'].notna().any():
        df_original_with_verdict = df_original
    else:
        df_original_with_verdict = add_verdict(
            df_original,
            id1_col=id1_col,
            id2_col=id2_col,
            conclusion_col=conclusion_col,
            positive_label=positive_label,
            new_col='verdict'
        )
    
    # Create a set of bidirectionally entailed pairs (verdict = YES)
    bidirectional_pairs = set()
    for _, row in df_original_with_verdict[df_original_with_verdict['verdict'] == positive_label].iterrows():
        # Create sorted tuple so (A,B) == (B,A)
        pair = tuple(sorted([row[id1_col], row[id2_col]]))
        bidirectional_pairs.add(pair)
    
    # Check pairs in df
    total_pairs = len(df)
    entailed_pairs_count = 0
    
    for _, row in df.iterrows():
        # Create sorted tuple to match bidirectional_pairs format
        pair = tuple(sorted([row[id1_col], row[id2_col]]))
        
        # Check if it's bidirectionally entailed
        if pair in bidirectional_pairs:
            entailed_pairs_count += 1
    
    # Calculate ratio
    entailment_ratio = entailed_pairs_count / total_pairs if total_pairs > 0 else 0.0
    
    return {
        'total_pairs': total_pairs,
        'bidirectionally_entailed': entailed_pairs_count,
        'entailment_ratio': entailment_ratio
    }


# ================================================================
# LLM LABELED PAIRS TRACKING & BIDIRECTIONAL PROCESSING
# ================================================================

def append_to_llm_labeled_pairs(
    df_pairs: pd.DataFrame,
    labeled_csv: str = "labeled_pairs/llm_labeled_pairs.csv",
    id1_col: str = 'sentence_id_1',
    id2_col: str = 'sentence_id_2',
) -> pd.DataFrame:
    """
    Append pairs to the master LLM-labeled-pairs CSV.

    Every pair we ever send to the LLM is recorded here so that
    :func:`generate_valid_pairs` can exclude them in future rounds.
    Duplicates (same id1, id2) are dropped.

    Parameters
    ----------
    df_pairs : pd.DataFrame
        New pairs to record.  Must contain ``id1_col`` and ``id2_col``.
    labeled_csv : str
        Path to the master CSV file (created if it doesn't exist).
    id1_col, id2_col : str
        Column names for the pair IDs.

    Returns
    -------
    pd.DataFrame
        The full accumulated labeled-pairs DataFrame.
    """
    os.makedirs(os.path.dirname(labeled_csv), exist_ok=True)

    df_new = df_pairs[[id1_col, id2_col]].copy()
    df_new = df_new.rename(columns={id1_col: 'sentence_id_1', id2_col: 'sentence_id_2'})

    if os.path.exists(labeled_csv):
        df_existing = pd.read_csv(labeled_csv)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all = df_all.drop_duplicates(subset=['sentence_id_1', 'sentence_id_2'])
    df_all.to_csv(labeled_csv, index=False)

    print(f"✓ LLM labeled pairs updated: {len(df_all)} total in {labeled_csv}")
    return df_all


def process_one_way_results(
    df_one_way: pd.DataFrame,
    id1_col: str = 'sentence_id_1',
    id2_col: str = 'sentence_id_2',
    conclusion_col: str = 'llm_conclusion_12',
    positive_label: str = 'YES',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process one-way LLM results into bidirectional verdicts.

    Given a df with one-way entailment conclusions (A→B only), this
    function identifies:

    1. **Resolved pairs** – pairs where both (A,B) and (B,A) appear in
       the df, so we can immediately determine the bidirectional verdict:
       - Both YES  → add to results with verdict YES  (both rows kept)
       - At least one NO → add to results with verdict NO (both rows kept)
    2. **Needs-reverse pairs** – pairs (A,B) that got YES but whose
       reverse (B,A) is **not** in the current df.  These must be sent
       back to the LLM for the second direction.

    **NO-verdict shortcut**: If A→B = NO the pair verdict is NO regardless
    of B→A.  We therefore immediately record *both* (A,B) and (B,A) with
    ``verdict='NO'`` so neither direction is ever sent to the LLM again.

    Parameters
    ----------
    df_one_way : pd.DataFrame
        LLM results with one-way conclusions.
    id1_col, id2_col : str
        Column names for pair IDs.
    conclusion_col : str
        Column with the LLM conclusion (YES/NO).
    positive_label : str
        Value indicating entailment.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        ``(df_resolved, df_needs_reverse)``

        - ``df_resolved``: pairs whose bidirectional verdict is known.
          Has all original columns plus a ``verdict`` column.
          Each undirected pair may produce **two** rows (A,B) and (B,A)
          so that both directions are recorded.
        - ``df_needs_reverse``: pairs that need to be sent to the LLM
          in the reverse direction.  Contains the original row columns
          plus ``reverse_id1`` and ``reverse_id2`` indicating which
          direction to query.
    """
    df = df_one_way.copy()

    # Build a lookup: (id1, id2) -> conclusion
    pair_lookup = {}
    for _, row in df.iterrows():
        key = (str(row[id1_col]), str(row[id2_col]))
        pair_lookup[key] = str(row[conclusion_col]).strip().upper()

    resolved_rows = []
    needs_reverse_rows = []

    # Track pairs we've already resolved (via their sorted key) to avoid
    # processing (A,B) and (B,A) separately when both appear in df.
    resolved_sorted_keys = set()

    def _make_reverse_row(orig_row, verdict_val):
        """Create a synthetic reverse-direction row (B,A) from an (A,B) row."""
        rev = orig_row.copy()
        rev[id1_col] = str(orig_row[id2_col])
        rev[id2_col] = str(orig_row[id1_col])
        # Swap text columns if present
        for t1, t2 in [('sentence_text_1', 'sentence_text_2'),
                        ('text1', 'text2')]:
            if t1 in rev.index and t2 in rev.index:
                rev[t1], rev[t2] = orig_row[t2], orig_row[t1]
        # Swap argument IDs if present
        if 'argument_id_1' in rev.index and 'argument_id_2' in rev.index:
            rev['argument_id_1'], rev['argument_id_2'] = (
                orig_row['argument_id_2'], orig_row['argument_id_1'])
        rev['verdict'] = verdict_val
        # Mark the reverse conclusion as inferred (not from LLM)
        if conclusion_col in rev.index:
            rev[conclusion_col] = 'NO (inferred)'
        return rev

    for idx, row in df.iterrows():
        a = str(row[id1_col])
        b = str(row[id2_col])
        fwd_conclusion = str(row[conclusion_col]).strip().upper()
        sorted_key = tuple(sorted([a, b]))

        if sorted_key in resolved_sorted_keys:
            continue  # Already handled as part of the reverse pair

        rev_key = (b, a)
        rev_conclusion = pair_lookup.get(rev_key, None)

        if fwd_conclusion != positive_label.upper():
            # Forward is NO → pair is NO regardless of reverse
            # Record (A,B) = NO
            resolved_row = row.copy()
            resolved_row['verdict'] = 'NO'
            resolved_rows.append(resolved_row)
            # NO-verdict shortcut: also record (B,A) = NO
            resolved_rows.append(_make_reverse_row(row, 'NO'))
            resolved_sorted_keys.add(sorted_key)
        elif rev_conclusion is not None:
            # Both directions exist in df
            if rev_conclusion == positive_label.upper():
                # Both YES → verdict YES (record both directions)
                fwd_row = row.copy()
                fwd_row['verdict'] = 'YES'
                resolved_rows.append(fwd_row)
                # Find the actual reverse row in df for the (B,A) entry
                rev_mask = (df[id1_col].astype(str) == b) & (df[id2_col].astype(str) == a)
                if rev_mask.any():
                    rev_row = df.loc[rev_mask.idxmax()].copy()
                    rev_row['verdict'] = 'YES'
                    resolved_rows.append(rev_row)
            else:
                # One is NO → verdict NO (record both directions)
                fwd_row = row.copy()
                fwd_row['verdict'] = 'NO'
                resolved_rows.append(fwd_row)
                rev_mask = (df[id1_col].astype(str) == b) & (df[id2_col].astype(str) == a)
                if rev_mask.any():
                    rev_row = df.loc[rev_mask.idxmax()].copy()
                    rev_row['verdict'] = 'NO'
                    resolved_rows.append(rev_row)
            resolved_sorted_keys.add(sorted_key)
        else:
            # Forward is YES but reverse doesn't exist → need to query LLM
            needs_reverse_row = row.copy()
            needs_reverse_row['reverse_id1'] = b
            needs_reverse_row['reverse_id2'] = a
            needs_reverse_rows.append(needs_reverse_row)
            resolved_sorted_keys.add(sorted_key)

    df_resolved = pd.DataFrame(resolved_rows)
    df_needs_reverse = pd.DataFrame(needs_reverse_rows)

    n_yes = (df_resolved['verdict'] == 'YES').sum() if len(df_resolved) > 0 else 0
    n_no = (df_resolved['verdict'] == 'NO').sum() if len(df_resolved) > 0 else 0
    n_inferred = 0
    if len(df_resolved) > 0 and conclusion_col in df_resolved.columns:
        n_inferred = (df_resolved[conclusion_col].astype(str).str.contains('inferred', case=False)).sum()

    print(f"\n{'='*60}")
    print(f"ONE-WAY RESULTS PROCESSING")
    print(f"{'='*60}")
    print(f"Total input pairs: {len(df)}")
    print(f"Resolved rows: {len(df_resolved)} (YES={n_yes}, NO={n_no})")
    print(f"  ↳ Inferred reverse NO (money saved): {n_inferred}")
    print(f"Need reverse LLM call: {len(df_needs_reverse)}")
    print(f"{'='*60}\n")

    return df_resolved, df_needs_reverse


def send_back_to_llm(
    df_needs_reverse: pd.DataFrame,
    df_clause: pd.DataFrame,
    model: str = "deepseek-reasoner",
    prompt_type: str = "test_prompt_tot_json2",
    args_file: str = "ArgLevel_ClauseIds_df.xlsx",
    output_dir: str = "labeled_pairs",
    batch_label: str = "reverse_check",
    save_every_n: int = 500,
    deepseek_api_key: Optional[str] = None,
    positive_label: str = 'YES',
) -> pd.DataFrame:
    """
    Send the reverse direction of pairs to the LLM and combine results.

    Takes pairs from :func:`process_one_way_results` that had YES in the
    forward direction but whose reverse wasn't available, queries the LLM
    for the reverse, and returns resolved bidirectional results.

    Parameters
    ----------
    df_needs_reverse : pd.DataFrame
        Output from :func:`process_one_way_results` with columns
        ``reverse_id1``, ``reverse_id2``, plus the original pair columns.
    df_clause : pd.DataFrame
        Clause-level DataFrame with ``sentence_id`` and ``sentence`` (or
        ``sentence_text``) columns, used to look up text for reverse pairs.
    model : str
        LLM model name.
    prompt_type : str
        Prompt template key.
    args_file : str
        Path to argument-level context Excel file.
    output_dir : str
        Directory for saving intermediate results.
    batch_label : str
        Label for intermediate files.
    save_every_n : int
        Save intermediate results every N pairs.
    deepseek_api_key : str or None
        API key (set as env var if provided).
    positive_label : str
        Value indicating entailment.

    Returns
    -------
    pd.DataFrame
        All reverse pairs with their bidirectional verdict resolved.
        Has columns matching the original one-way results plus ``verdict``.
    """
    import sys

    if len(df_needs_reverse) == 0:
        print("No reverse pairs to send to LLM.")
        return pd.DataFrame()

    if deepseek_api_key:
        os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key

    # Build the reverse-direction input DataFrame for the evaluator.
    # The evaluator expects: sentence_id_1, sentence_id_2, sentence_text_1,
    #   sentence_text_2, argument_id_1, argument_id_2
    # We look up text from df_clause (keyed by sentence_id → sentence).
    # Support both 'sentence' and 'sentence_text' column names.
    text_col = 'sentence_text' if 'sentence_text' in df_clause.columns else 'sentence'
    text_lookup = df_clause.set_index('sentence_id')[text_col].to_dict()

    rev_id1 = df_needs_reverse['reverse_id1'].values
    rev_id2 = df_needs_reverse['reverse_id2'].values

    df_reverse_input = pd.DataFrame({
        'sentence_id_1': rev_id1,
        'sentence_id_2': rev_id2,
        'sentence_text_1': [text_lookup.get(str(sid), '') for sid in rev_id1],
        'sentence_text_2': [text_lookup.get(str(sid), '') for sid in rev_id2],
    })

    # Add argument IDs
    df_reverse_input['argument_id_1'] = df_reverse_input['sentence_id_1'].apply(extract_argument_id)
    df_reverse_input['argument_id_2'] = df_reverse_input['sentence_id_2'].apply(extract_argument_id)

    # Save to temp CSV for evaluator
    os.makedirs(output_dir, exist_ok=True)
    reverse_input_csv = os.path.join(output_dir, f"{batch_label}_input.csv")
    df_reverse_input.to_csv(reverse_input_csv, index=False)

    # Also save intermediate LLM results every N pairs
    intermediate_dir = os.path.join(output_dir, "llm_intermediates")
    os.makedirs(intermediate_dir, exist_ok=True)

    reverse_output_base = os.path.join(output_dir, f"{batch_label}_output")

    print(f"\n{'='*60}")
    print(f"SENDING {len(df_reverse_input)} REVERSE PAIRS TO LLM")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Input: {reverse_input_csv}")
    print(f"Output: {reverse_output_base}.csv")

    # Import and run the evaluator
    import importlib
    import llm_calls.deepseek_evaluator as etb
    importlib.reload(etb)

    sys.argv = [
        "deepseek_evaluator.py",
        "--model", model,
        "--file", reverse_input_csv,
        "--external", args_file,
        "--prompt", prompt_type,
        "--output", reverse_output_base,
    ]

    etb.main()
    print(f"✓ Reverse LLM evaluation complete")

    # Read reverse results
    reverse_output_csv = f"{reverse_output_base}.csv"
    df_reverse_results = pd.read_csv(reverse_output_csv)

    # Record these in llm_labeled_pairs
    append_to_llm_labeled_pairs(df_reverse_results)

    # Now resolve: the forward was YES. If reverse is YES → verdict YES, else NO.
    # Produce BOTH directional rows (A,B) and (B,A) for each pair.
    resolved_rows = []
    reverse_lookup = {}
    for _, row in df_reverse_results.iterrows():
        key = (str(row['sentence_id_1']), str(row['sentence_id_2']))
        conclusion = str(row.get('llm_conclusion_12', '')).strip().upper()
        reverse_lookup[key] = conclusion

    id1_col = 'sentence_id_1'
    id2_col = 'sentence_id_2'

    for _, orig_row in df_needs_reverse.iterrows():
        rev_id1 = str(orig_row['reverse_id1'])
        rev_id2 = str(orig_row['reverse_id2'])
        rev_conclusion = reverse_lookup.get((rev_id1, rev_id2), 'NO')

        verdict = 'YES' if rev_conclusion == positive_label.upper() else 'NO'

        # Forward row (A,B)
        fwd_row = orig_row.drop(labels=['reverse_id1', 'reverse_id2'], errors='ignore').copy()
        fwd_row['verdict'] = verdict
        resolved_rows.append(fwd_row)

        # Reverse row (B,A) — use the actual LLM result row if available,
        # otherwise construct from forward row.
        rev_key = (rev_id1, rev_id2)
        if rev_key in reverse_lookup:
            # Find matching row in df_reverse_results
            rev_mask = (
                (df_reverse_results['sentence_id_1'].astype(str) == rev_id1) &
                (df_reverse_results['sentence_id_2'].astype(str) == rev_id2)
            )
            if rev_mask.any():
                rev_row = df_reverse_results.loc[rev_mask.idxmax()].copy()
                rev_row['verdict'] = verdict
                resolved_rows.append(rev_row)
            else:
                # Fallback: construct synthetic reverse row
                rev_row = fwd_row.copy()
                rev_row[id1_col] = rev_id1
                rev_row[id2_col] = rev_id2
                for t1, t2 in [('sentence_text_1', 'sentence_text_2')]:
                    if t1 in rev_row.index and t2 in rev_row.index:
                        rev_row[t1], rev_row[t2] = fwd_row.get(t2, ''), fwd_row.get(t1, '')
                if 'argument_id_1' in rev_row.index and 'argument_id_2' in rev_row.index:
                    rev_row['argument_id_1'], rev_row['argument_id_2'] = (
                        fwd_row.get('argument_id_2', ''), fwd_row.get('argument_id_1', ''))
                rev_row['verdict'] = verdict
                resolved_rows.append(rev_row)
        else:
            # No LLM result for reverse → construct synthetic row
            rev_row = fwd_row.copy()
            rev_row[id1_col] = rev_id1
            rev_row[id2_col] = rev_id2
            rev_row['verdict'] = verdict
            resolved_rows.append(rev_row)

    df_final_resolved = pd.DataFrame(resolved_rows)

    n_yes = (df_final_resolved['verdict'] == 'YES').sum() if len(df_final_resolved) > 0 else 0
    n_no = (df_final_resolved['verdict'] == 'NO').sum() if len(df_final_resolved) > 0 else 0

    print(f"\n{'='*60}")
    print(f"REVERSE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total reverse pairs resolved: {len(df_final_resolved)}")
    print(f"  Bidirectional YES: {n_yes}")
    print(f"  Bidirectional NO:  {n_no}")
    print(f"{'='*60}\n")

    return df_final_resolved


def process_llm_results_bidirectional(
    df_one_way: pd.DataFrame,
    df_clause: pd.DataFrame,
    results_output_path: str,
    model: str = "deepseek-reasoner",
    prompt_type: str = "test_prompt_tot_json2",
    args_file: str = "ArgLevel_ClauseIds_df.xlsx",
    output_dir: str = "labeled_pairs",
    batch_label: str = "reverse_check",
    save_every_n: int = 500,
    deepseek_api_key: Optional[str] = None,
    labeled_csv: str = "labeled_pairs/llm_labeled_pairs.csv",
    max_reverse_pairs: int = 100_000,
) -> pd.DataFrame:
    """
    Full pipeline: take one-way LLM results, resolve bidirectional verdicts,
    send reverse queries as needed, save everything.

    Steps:
    1. Record all input pairs in ``llm_labeled_pairs.csv``.
    2. Process one-way results to find resolved and needs-reverse pairs.
    3. Send reverse pairs to LLM (capped at ``max_reverse_pairs``).
    4. Record reverse pairs in ``llm_labeled_pairs.csv``.
    5. Combine all resolved pairs and save to ``results_output_path``.

    Parameters
    ----------
    df_one_way : pd.DataFrame
        Raw one-way LLM output.
    df_clause : pd.DataFrame
        Clause-level DataFrame with ``sentence_id`` and ``sentence`` (or
        ``sentence_text``) columns, used to look up text for reverse pairs.
    results_output_path : str
        Where to save the final bidirectional results CSV.
    model, prompt_type, args_file, output_dir, batch_label, save_every_n
        Passed to :func:`send_back_to_llm`.
    deepseek_api_key : str or None
        API key.
    labeled_csv : str
        Path to master labeled-pairs CSV.
    max_reverse_pairs : int
        Cap on reverse pairs to send to LLM per call.

    Returns
    -------
    pd.DataFrame
        Final bidirectional results DataFrame.
    """
    # Step 1: Record all one-way pairs in llm_labeled_pairs
    append_to_llm_labeled_pairs(df_one_way, labeled_csv=labeled_csv)

    # Step 2: Separate resolved vs needs-reverse
    # (process_one_way_results now produces BOTH directional rows for
    #  NO-verdict pairs — e.g. if AB=NO, both (A,B) and (B,A) are
    #  added to df_resolved with verdict=NO)
    df_resolved, df_needs_reverse = process_one_way_results(df_one_way)

    # Step 2.1: Record the inferred reverse-NO pairs in llm_labeled_pairs
    # so they are never sent to the LLM in future iterations.
    if len(df_resolved) > 0 and 'llm_conclusion_12' in df_resolved.columns:
        inferred_mask = df_resolved['llm_conclusion_12'].astype(str).str.contains('inferred', case=False, na=False)
        if inferred_mask.any():
            df_inferred = df_resolved[inferred_mask]
            append_to_llm_labeled_pairs(df_inferred, labeled_csv=labeled_csv)
            print(f"✓ Recorded {len(df_inferred)} inferred reverse-NO pairs in {labeled_csv}")

    # Step 2.5: Filter out reverse pairs that were already sent to LLM
    # IMPORTANT: Use DIRECTIONAL check — (B,A) is only "already labeled" if
    # (B,A) itself was sent, NOT if (A,B) was sent (entailment is directional).
    if len(df_needs_reverse) > 0 and 'reverse_id1' in df_needs_reverse.columns:
        before = len(df_needs_reverse)
        if os.path.exists(labeled_csv):
            df_labeled_master = pd.read_csv(labeled_csv)
            # Vectorised directional anti-join
            df_rev_keys = df_needs_reverse[['reverse_id1', 'reverse_id2']].copy()
            df_rev_keys.columns = ['sentence_id_1', 'sentence_id_2']
            df_rev_keys['sentence_id_1'] = df_rev_keys['sentence_id_1'].astype(str)
            df_rev_keys['sentence_id_2'] = df_rev_keys['sentence_id_2'].astype(str)

            df_lab_keys = df_labeled_master[['sentence_id_1', 'sentence_id_2']].drop_duplicates()
            df_lab_keys['sentence_id_1'] = df_lab_keys['sentence_id_1'].astype(str)
            df_lab_keys['sentence_id_2'] = df_lab_keys['sentence_id_2'].astype(str)
            df_lab_keys['__sent__'] = True

            check = df_rev_keys.merge(df_lab_keys, on=['sentence_id_1', 'sentence_id_2'], how='left')
            already_sent_mask = check['__sent__'].notna().values
            df_needs_reverse = df_needs_reverse[~already_sent_mask].copy()

            removed = before - len(df_needs_reverse)
            if removed > 0:
                print(f"✓ Filtered {removed} reverse pairs already sent (directional check)")

    # Step 3: Send reverse pairs to LLM (with cap)
    if len(df_needs_reverse) > max_reverse_pairs:
        print(f"⚠ Capping reverse pairs from {len(df_needs_reverse)} to {max_reverse_pairs}")
        df_needs_reverse = df_needs_reverse.head(max_reverse_pairs)

    if len(df_needs_reverse) > 0:
        df_reverse_resolved = send_back_to_llm(
            df_needs_reverse,
            df_clause=df_clause,
            model=model,
            prompt_type=prompt_type,
            args_file=args_file,
            output_dir=output_dir,
            batch_label=batch_label,
            save_every_n=save_every_n,
            deepseek_api_key=deepseek_api_key,
        )
    else:
        df_reverse_resolved = pd.DataFrame()

    # Step 4: Combine all resolved pairs
    all_resolved = pd.concat([df_resolved, df_reverse_resolved], ignore_index=True)

    # Step 5: Save to results file
    os.makedirs(os.path.dirname(results_output_path), exist_ok=True)
    all_resolved.to_csv(results_output_path, index=False)

    n_yes = (all_resolved['verdict'] == 'YES').sum() if len(all_resolved) > 0 else 0
    n_no = (all_resolved['verdict'] == 'NO').sum() if len(all_resolved) > 0 else 0

    print(f"\n{'='*60}")
    print(f"BIDIRECTIONAL PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total resolved pairs: {len(all_resolved)}")
    print(f"  YES (bidirectional): {n_yes}")
    print(f"  NO:                  {n_no}")
    print(f"Saved to: {results_output_path}")
    print(f"{'='*60}\n")

    return all_resolved


def filter_already_labeled(
    df_pairs: pd.DataFrame,
    labeled_csv: str = "labeled_pairs/llm_labeled_pairs.csv",
    id1_col: str = 'id1',
    id2_col: str = 'id2',
) -> pd.DataFrame:
    """
    Remove pairs from ``df_pairs`` that already appear in the master
    LLM-labeled-pairs CSV. Checks both directions: (A,B) and (B,A).

    Parameters
    ----------
    df_pairs : pd.DataFrame
        Candidate pairs to filter.
    labeled_csv : str
        Path to the master labeled-pairs CSV.
    id1_col, id2_col : str
        Column names for pair IDs in ``df_pairs``.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with already-labeled pairs removed.
    """
    if not os.path.exists(labeled_csv):
        print(f"No labeled pairs file found at {labeled_csv} — no filtering applied.")
        return df_pairs

    df_labeled = pd.read_csv(labeled_csv)

    # Build sorted-key columns for both dataframes (checks both directions)
    lab_a = df_labeled['sentence_id_1'].astype(str)
    lab_b = df_labeled['sentence_id_2'].astype(str)
    df_labeled_keys = pd.DataFrame({
        '_k1': np.where(lab_a <= lab_b, lab_a, lab_b),
        '_k2': np.where(lab_a <= lab_b, lab_b, lab_a),
    }).drop_duplicates()
    df_labeled_keys['__labeled__'] = True

    pair_a = df_pairs[id1_col].astype(str)
    pair_b = df_pairs[id2_col].astype(str)
    df_pairs = df_pairs.copy()
    df_pairs['_k1'] = np.where(pair_a <= pair_b, pair_a, pair_b)
    df_pairs['_k2'] = np.where(pair_a <= pair_b, pair_b, pair_a)

    merged = df_pairs.merge(df_labeled_keys, on=['_k1', '_k2'], how='left')
    df_filtered = merged[merged['__labeled__'].isna()].drop(columns=['_k1', '_k2', '__labeled__']).copy()
    df_filtered.reset_index(drop=True, inplace=True)

    # Also drop temp columns from the original if it was modified
    df_pairs.drop(columns=['_k1', '_k2'], inplace=True, errors='ignore')

    removed = len(df_pairs) - len(df_filtered)
    print(f"Filtered out {removed:,} already-labeled pairs (kept {len(df_filtered):,} of {len(df_pairs):,})")

    return df_filtered


# ================================================================
# PIPELINE / LOOP HELPER FUNCTIONS
# ================================================================

def load_pipeline_data(
    df_clause_path: Optional[str],
    embedding_cache_path: Optional[str],
    test: bool,
    remaining_llm_calls_path: Optional[str] = None,
    unlabeled_pairs_path: Optional[str] = None,
    iteration_number: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Load all pipeline data from pickle files for a single FEA Pipeline iteration.

    Parameters
    ----------
    df_clause_path : str or None
        Path to pickled df_clause.
    embedding_cache_path : str or None
        Path to pickled embedding cache.
    test : bool
        Whether running in test mode.
    remaining_llm_calls_path : str or None
        Path to pickled remaining LLM calls (test mode).
    unlabeled_pairs_path : str or None
        Path to pickled unlabeled pairs (production mode).
    iteration_number : int or None
        Current iteration number (for logging).

    Returns
    -------
    dict
        Keys: ``df_clause``, ``embedding_cache``, ``remaining_llm_calls``,
        ``unlabeled_pairs``.
    """
    print(f"\n{'='*80}")
    print(f"PARAMETER VALUES AFTER PAPERMILL INJECTION:")
    print(f"{'='*80}")
    if iteration_number is not None:
        print(f"iteration_number = {iteration_number}")
    print(f"test = {test}")
    print(f"remaining_llm_calls_path = {remaining_llm_calls_path}")
    print(f"df_clause_path = {df_clause_path}")
    print(f"{'='*80}\n")

    if df_clause_path:
        df_clause = pd.read_pickle(df_clause_path)
        print(f"✓ Loaded df_clause: {len(df_clause)} rows")
    else:
        raise ValueError("df_clause_path is None - papermill didn't inject parameters correctly")

    if embedding_cache_path:
        with open(embedding_cache_path, 'rb') as f:
            embedding_cache = pickle.load(f)
        print(f"✓ Loaded embedding cache: {len(embedding_cache)} embeddings")
    else:
        raise ValueError("embedding_cache_path is None - papermill didn't inject parameters correctly")

    remaining_llm_calls = None
    unlabeled_pairs = None

    if test and remaining_llm_calls_path:
        remaining_llm_calls = pd.read_pickle(remaining_llm_calls_path)
        print(f"✓ Loaded remaining_llm_calls: {len(remaining_llm_calls)} rows")
    elif not test and unlabeled_pairs_path:
        unlabeled_pairs = pd.read_pickle(unlabeled_pairs_path)
        print(f"✓ Loaded unlabeled_pairs: {len(unlabeled_pairs)} rows")
    else:
        print(f"⚠ WARNING: No data loaded!")
        print(f"  test={test}, remaining_llm_calls_path={remaining_llm_calls_path}, unlabeled_pairs_path={unlabeled_pairs_path}")

    print(f"✓ All data loaded from pickle files")

    return {
        'df_clause': df_clause,
        'embedding_cache': embedding_cache,
        'remaining_llm_calls': remaining_llm_calls,
        'unlabeled_pairs': unlabeled_pairs,
    }


def run_fea_papermill(
    iteration_number: int,
    df_candidates: pd.DataFrame = None,
    df_crossed: pd.DataFrame = None,
    df_labeled: pd.DataFrame = None,
    df_labeled_crossed: pd.DataFrame = None,
    df_obs_ent: pd.DataFrame = None,
    df_clause: pd.DataFrame = None,
    embedding_cache: dict = None,
    temp_dir: str = "fea_iterations/temp_data",
    data_on_disk: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Run FreeEntailmentAlgorithm.ipynb via papermill and retrieve results.

    When ``data_on_disk=False`` (default, backward-compatible), saves all
    DataFrames to disk first.  When ``data_on_disk=True``, assumes the
    caller has already pickled everything to *temp_dir* and freed memory.

    Parameters
    ----------
    iteration_number : int
        Current iteration index.
    df_candidates, df_crossed, df_labeled, df_labeled_crossed, df_obs_ent, df_clause
        DataFrames produced by the FEA Pipeline (ignored when *data_on_disk*).
    embedding_cache : dict
        Pre-computed embedding cache (ignored when *data_on_disk*).
    temp_dir : str
        Directory for temporary pickle files.
    data_on_disk : bool, default False
        If True, skip pickling — the caller already saved everything to
        *temp_dir* and freed the large DataFrames to reclaim RAM before
        FreeEntailmentAlgorithm spawns a new process.

    Returns
    -------
    (pd.DataFrame, str)
        ``(df_final, fig_html)``
    """
    import papermill as pm
    import scrapbook as sb

    os.makedirs(temp_dir, exist_ok=True)

    if not data_on_disk:
        # Short-circuit: if df_candidates or df_obs_ent are empty, there's nothing
        # for FreeEntailmentAlgorithm to train on — return empty results immediately.
        if len(df_candidates) == 0 or len(df_obs_ent) == 0:
            print(f"⚠ Skipping FreeEntailmentAlgorithm (empty data: "
                  f"{len(df_candidates)} candidates, {len(df_obs_ent)} entailed pairs)")
            df_final = pd.DataFrame(columns=['id1', 'id2', 'text1', 'text2', 'entailment_probability'])
            fig_html = "<p>No data for this iteration</p>"
            return df_final, fig_html

        df_candidates.to_pickle(f"{temp_dir}/df_candidates.pkl")
        print(f"  df_candidates pickled: {len(df_candidates):,} rows, "
              f"cols={list(df_candidates.columns)}")
        df_crossed.to_pickle(f"{temp_dir}/df_crossed.pkl")
        df_labeled.to_pickle(f"{temp_dir}/df_labeled.pkl")
        df_labeled_crossed.to_pickle(f"{temp_dir}/df_labeled_crossed.pkl")
        df_obs_ent.to_pickle(f"{temp_dir}/df_obs_ent.pkl")
        df_clause.to_pickle(f"{temp_dir}/df_clause.pkl")

        if embedding_cache is not None:
            with open(f"{temp_dir}/embedding_cache.pkl", 'wb') as f:
                pickle.dump(embedding_cache, f)

    # Only pass embedding_cache_path if the file exists (cosine sims may
    # already be pre-computed in the DataFrames by Pipeline).
    _emb_cache_pkl = f"{temp_dir}/embedding_cache.pkl"
    parameters = {
        "df_candidates_path": f"{temp_dir}/df_candidates.pkl",
        "df_crossed_path": f"{temp_dir}/df_crossed.pkl",
        "df_labeled_path": f"{temp_dir}/df_labeled.pkl",
        "df_labeled_crossed_path": f"{temp_dir}/df_labeled_crossed.pkl",
        "df_obs_ent_path": f"{temp_dir}/df_obs_ent.pkl",
        "df_clause_path": f"{temp_dir}/df_clause.pkl",
        "embedding_cache_path": _emb_cache_pkl if os.path.exists(_emb_cache_pkl) else "",
    }

    output_notebook_path = f"fea_iterations/FEA_iter_{iteration_number}.ipynb"
    os.makedirs("fea_iterations", exist_ok=True)

    print(f"Executing FreeEntailmentAlgorithm.ipynb for iteration {iteration_number}...")
    pm.execute_notebook(
        'FreeEntailmentAlgorithm.ipynb',
        output_notebook_path,
        parameters=parameters,
    )

    # Read outputs from pickle files (NOT scrapbook — scrapbook's JSON
    # serialization through Jupyter messaging OOMs at 75 M-candidate scale).
    df_final_path = f"{temp_dir}/df_final.pkl"
    fig_html_path = f"{temp_dir}/fig_html.pkl"

    if os.path.exists(df_final_path):
        df_final = pd.read_pickle(df_final_path)
    else:
        # Fallback to scrapbook if pickle wasn't written (shouldn't happen)
        nb = sb.read_notebook(output_notebook_path)
        df_final = nb.scraps['df_final'].data

    if os.path.exists(fig_html_path):
        with open(fig_html_path, 'rb') as f:
            fig_html = pickle.load(f)
    else:
        if 'nb' not in locals():
            nb = sb.read_notebook(output_notebook_path)
        fig_html = nb.scraps.get('fig_html', type('', (), {'data': '<p>Plot unavailable</p>'})).data

    print(f"✓ Retrieved outputs:")
    print(f"  - df_final: {len(df_final)} rows")
    print(f"  - fig_html: HTML plot ({len(fig_html)} chars)")

    return df_final, fig_html


def extract_argument_id(sentence_id: str) -> Optional[str]:
    """
    Extract argument ID from a sentence ID based on its prefix.

    - Speech IDs (``SXXXXX...``) → first 6 characters (e.g. ``S11150``)
    - Book IDs (``BXXXX...``)  → first 5 characters (e.g. ``B0249``)

    Parameters
    ----------
    sentence_id : str
        Full sentence ID string.

    Returns
    -------
    str or None
    """
    if not isinstance(sentence_id, str):
        return None
    if sentence_id.startswith('S'):
        return sentence_id[:8] if len(sentence_id) >= 8 else sentence_id
    elif sentence_id.startswith('B'):
        return sentence_id[:5] if len(sentence_id) >= 5 else sentence_id
    return None


def format_df_to_llm(
    df_to_llm: pd.DataFrame,
    df_clause: Optional[pd.DataFrame] = None,
    id_col: str = 'sentence_id',
    text_col: str = 'sentence',
) -> pd.DataFrame:
    """
    Format a DataFrame of FEA-selected pairs for LLM evaluation.

    Steps:
    1. Keep only ``[id1, id2, text1, text2, entailment_probability]``.
    2. Rename columns to match the LLM input schema.
    3. Add ``argument_id_1`` / ``argument_id_2`` via :func:`extract_argument_id`.
    4. Reorder columns.

    Parameters
    ----------
    df_to_llm : pd.DataFrame
        Must contain ``id1, id2, entailment_probability``.
        ``text1, text2`` are added automatically from ``df_clause``
        if not already present.
    df_clause : pd.DataFrame, optional
        Clause-level DataFrame used to look up text when text columns
        are missing from ``df_to_llm``.
    id_col, text_col : str
        Column names in ``df_clause``.

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with columns:
        ``[sentence_id_2, sentence_id_1, sentence_text_2, argument_id_2,
        sentence_text_1, argument_id_1, score]``
    """
    # Lazy text lookup
    if df_clause is not None:
        _lut = dict(zip(df_clause[id_col].astype(str), df_clause[text_col]))
        if 'text1' not in df_to_llm.columns:
            df_to_llm = df_to_llm.copy()
            df_to_llm['text1'] = df_to_llm['id1'].astype(str).map(_lut)
        if 'text2' not in df_to_llm.columns:
            df_to_llm = df_to_llm.copy()
            df_to_llm['text2'] = df_to_llm['id2'].astype(str).map(_lut)

    df = df_to_llm[['id1', 'id2', 'text1', 'text2', 'entailment_probability']].copy()

    df = df.rename(columns={
        'id1': 'sentence_id_1',
        'id2': 'sentence_id_2',
        'text1': 'sentence_text_1',
        'text2': 'sentence_text_2',
        'entailment_probability': 'score',
    })

    df['argument_id_1'] = df['sentence_id_1'].apply(extract_argument_id)
    df['argument_id_2'] = df['sentence_id_2'].apply(extract_argument_id)

    df = df[['sentence_id_2', 'sentence_id_1', 'sentence_text_2', 'argument_id_2',
             'sentence_text_1', 'argument_id_1', 'score']]

    return df


def finalize_pipeline_iteration(
    test: bool,
    df_to_llm: pd.DataFrame,
    iteration_number: int,
    remaining_llm_calls: Optional[pd.DataFrame] = None,
    remaining_llm_calls_path: Optional[str] = None,
    unlabeled_pairs: Optional[pd.DataFrame] = None,
    unlabeled_pairs_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Handle end-of-iteration bookkeeping for FEA_Pipeline.

    In **test mode** the function merges ``df_to_llm`` with
    ``remaining_llm_calls`` to mock LLM results, removes matched pairs,
    and saves outputs.

    In **production mode** it saves ``df_to_llm`` for actual LLM processing,
    removes sent pairs from ``unlabeled_pairs``, and glues results to
    scrapbook.

    Parameters
    ----------
    test : bool
        Whether running in test mode.
    df_to_llm : pd.DataFrame
        Pairs selected for LLM evaluation (already formatted).
    iteration_number : int
        Current iteration index.
    remaining_llm_calls : pd.DataFrame or None
        Remaining mock LLM data (test mode).
    remaining_llm_calls_path : str or None
        Path to save updated remaining calls.
    unlabeled_pairs : pd.DataFrame or None
        Remaining unlabeled pairs (production mode).
    unlabeled_pairs_path : str or None
        Path to save updated unlabeled pairs.

    Returns
    -------
    dict
        ``remaining_llm_calls`` and ``unlabeled_pairs`` (updated or
        unchanged).
    """
    import scrapbook as sb

    output_csv = f"fea_iterations/llm_batch_iter_{iteration_number}.csv"

    if test:
        print(f"\n{'='*60}")
        print("TEST MODE: Mocking LLM responses")
        print(f"{'='*60}")

        df_to_llm_with_results = df_to_llm.merge(
            remaining_llm_calls[['sentence_id_1', 'sentence_id_2', 'answers_12',
                                  'reasonings_12', 'comment_12', 'llm_confidence_12',
                                  'llm_conclusion_12']],
            on=['sentence_id_1', 'sentence_id_2'],
            how='left',
        )

        matched = df_to_llm_with_results['llm_conclusion_12'].notna().sum()
        print(f"✓ Matched {matched}/{len(df_to_llm_with_results)} pairs with mock LLM results")
        if matched < len(df_to_llm_with_results):
            print(f"⚠ Warning: {len(df_to_llm_with_results) - matched} pairs have no mock LLM data")

        # Set-based anti-join for test mode
        remove_set = set(
            zip(
                df_to_llm['sentence_id_1'].astype(str),
                df_to_llm['sentence_id_2'].astype(str),
            )
        )
        before_n = len(remaining_llm_calls)
        n = len(remaining_llm_calls)
        mask = np.ones(n, dtype=bool)
        chunk_sz = 2_000_000
        for lo in range(0, n, chunk_sz):
            hi = min(lo + chunk_sz, n)
            ids1 = remaining_llm_calls['sentence_id_1'].iloc[lo:hi].astype(str).tolist()
            ids2 = remaining_llm_calls['sentence_id_2'].iloc[lo:hi].astype(str).tolist()
            for j in range(hi - lo):
                if (ids1[j], ids2[j]) in remove_set:
                    mask[lo + j] = False
            del ids1, ids2

        remaining_llm_calls = remaining_llm_calls[mask]
        remaining_llm_calls.reset_index(drop=True, inplace=True)
        removed = before_n - len(remaining_llm_calls)

        print(f"✓ Removed {removed} pairs from remaining LLM calls")
        print(f"✓ Remaining pairs for future iterations: {len(remaining_llm_calls)}")

        if remaining_llm_calls_path:
            remaining_llm_calls.to_pickle(remaining_llm_calls_path)
            print(f"✓ Saved updated remaining_llm_calls to {remaining_llm_calls_path}")

        df_to_llm_with_results.to_csv(output_csv, index=False)
        print(f"✓ Saved {len(df_to_llm_with_results)} pairs with LLM results to {output_csv}")
    else:
        df_to_llm.to_csv(output_csv, index=False)
        print(f"✓ Saved {len(df_to_llm)} pairs to {output_csv} for LLM processing")

        # Track in llm_labeled_pairs
        append_to_llm_labeled_pairs(df_to_llm)

        # Set-based anti-join — avoids creating a full 75M-row merge copy.
        # df_to_llm is tiny (≤ 1 k rows), so the set lookup is instant.
        remove_set = set(
            zip(
                df_to_llm['sentence_id_1'].astype(str),
                df_to_llm['sentence_id_2'].astype(str),
            )
        )
        # Map column names: df_to_llm uses sentence_id_1/2, unlabeled uses id1/id2
        before_n = len(unlabeled_pairs)
        n = len(unlabeled_pairs)
        mask = np.ones(n, dtype=bool)
        chunk_sz = 2_000_000
        for lo in range(0, n, chunk_sz):
            hi = min(lo + chunk_sz, n)
            ids1 = unlabeled_pairs['id1'].iloc[lo:hi].astype(str).tolist()
            ids2 = unlabeled_pairs['id2'].iloc[lo:hi].astype(str).tolist()
            for j in range(hi - lo):
                if (ids1[j], ids2[j]) in remove_set:
                    mask[lo + j] = False
            del ids1, ids2

        unlabeled_pairs = unlabeled_pairs[mask]
        unlabeled_pairs.reset_index(drop=True, inplace=True)
        removed = before_n - len(unlabeled_pairs)

        print(f"✓ Removed {removed} pairs from unlabeled_pairs")
        print(f"✓ Remaining pairs for future iterations: {len(unlabeled_pairs)}")

        if unlabeled_pairs_path:
            unlabeled_pairs.to_pickle(unlabeled_pairs_path)
            print(f"✓ Saved updated unlabeled_pairs to {unlabeled_pairs_path}")

        sb.glue('df_to_llm', df_to_llm)
        print(f"✓ Glued df_to_llm to scrapbook for FEA_Loop retrieval")

    print(f"\nIteration {iteration_number} complete")

    return {
        'remaining_llm_calls': remaining_llm_calls,
        'unlabeled_pairs': unlabeled_pairs,
    }


def run_fea_loop(
    test: bool,
    input_file: Union[str, pd.DataFrame],
    df_clause: pd.DataFrame,
    embedding_cache: dict,
    num_iterations: int,
    start_iteration: int,
    output_dir: str,
    sent_frac: float,
    budget: float,
    remaining_llm_calls: Optional[pd.DataFrame] = None,
    unlabeled_pairs: Optional[pd.DataFrame] = None,
    deepseek_api_key: Optional[str] = None,
    budget_dollars: Optional[float] = None,
    cost_per_1k_pairs: float = 2.0,
) -> List[dict]:
    """
    Run the full FEA iterative loop.

    Each iteration executes ``FEA_Pipeline.ipynb`` via papermill, and
    optionally runs the LLM evaluator notebook in production mode.

    The loop runs until one of three stopping conditions is met:

    1. ``num_iterations`` iterations have been completed.
    2. ``budget_dollars`` has been exhausted (production mode only).
    3. The unlabeled pool is empty.

    Parameters
    ----------
    test : bool
        Test mode flag.
    input_file : str or pd.DataFrame
        Initial labeled data (CSV path or DataFrame).
    df_clause : pd.DataFrame
        All clauses (premises + conclusions).
    embedding_cache : dict
        Pre-computed embedding cache.
    num_iterations : int
        Maximum loop iterations to run (hard cap).
    start_iteration : int
        Starting iteration index.
    output_dir : str
        Directory for output notebooks / data.
    sent_frac : float
        Fraction of df_final to send to LLM.
    budget : float
        Budget parameter passed to the pipeline.
    remaining_llm_calls : pd.DataFrame or None
        Mock LLM data for test mode.
    unlabeled_pairs : pd.DataFrame or None
        Unlabeled pairs for production mode.
    deepseek_api_key : str or None
        API key set as ``DEEPSEEK_API_KEY`` env var in production mode.
    budget_dollars : float or None
        Maximum dollar amount to spend on LLM calls.  When the cumulative
        cost reaches or exceeds this value the loop stops *before*
        starting a new iteration.  ``None`` means unlimited.
    cost_per_1k_pairs : float
        Estimated cost in dollars per 1,000 pairs sent to the LLM.
        Default is ``2.0`` ($2 / 1 k pairs).

    Returns
    -------
    list[dict]
        List of per-iteration statistics dictionaries.
    """
    import papermill as pm
    import scrapbook as sb

    # --- Validate initial data state ---
    print("=" * 60)
    print("VALIDATING INITIAL DATA STATE")
    print("=" * 60)

    if isinstance(input_file, pd.DataFrame):
        actual_initial_labeled = len(input_file)
    elif isinstance(input_file, str):
        actual_initial_labeled = len(pd.read_csv(input_file))
    else:
        raise ValueError(f"input_file has unexpected type: {type(input_file)}")

    if test:
        actual_initial_unlabeled = len(remaining_llm_calls) if remaining_llm_calls is not None else 0
    else:
        actual_initial_unlabeled = len(unlabeled_pairs) if unlabeled_pairs is not None else 0

    print(f"Initial labeled pairs: {actual_initial_labeled}")
    print(f"Initial unlabeled pairs: {actual_initial_unlabeled}")
    print(f"Total: {actual_initial_labeled + actual_initial_unlabeled}")

    loop_data_dir = os.path.join("fea_iterations", "loop_data")
    if os.path.exists(loop_data_dir):
        shutil.rmtree(loop_data_dir)
        print(f"✓ Cleaned up old loop data directory")

    iteration_stats: List[dict] = []
    total_pairs = actual_initial_labeled + actual_initial_unlabeled
    cumulative_labeled = actual_initial_labeled
    unlabeled_pool = actual_initial_unlabeled

    print(f"Starting with {cumulative_labeled} already labeled pairs")
    print(f"Unlabeled pool: {unlabeled_pool}")
    print(f"Total pairs: {total_pairs}")

    cost_per_pair = cost_per_1k_pairs / 1000.0
    cost_spent = 0.0

    if budget_dollars is not None:
        print(f"Budget: ${budget_dollars:,.2f}  (${cost_per_1k_pairs:.2f} / 1k pairs)")
    else:
        print("Budget: unlimited")

    labeled_pairs_dir = "labeled_pairs"
    os.makedirs(labeled_pairs_dir, exist_ok=True)

    # --- Main loop ---
    for iteration in range(start_iteration, num_iterations + 1):
        # --- Budget gate: stop before starting a new iteration if budget exhausted ---
        if budget_dollars is not None and cost_spent >= budget_dollars:
            print(f"\n{'='*60}")
            print(f"BUDGET EXHAUSTED  (${cost_spent:,.2f} / ${budget_dollars:,.2f})")
            print(f"Stopping before iteration {iteration}.")
            print(f"{'='*60}")
            break

        remaining_budget_str = (
            f"${budget_dollars - cost_spent:,.2f} remaining"
            if budget_dollars is not None
            else "unlimited"
        )
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{num_iterations}")
        print(f"{'='*60}")
        print(f"Status: Labeled={cumulative_labeled}, Unlabeled={unlabeled_pool}, Total={total_pairs}")
        print(f"Cost so far: ${cost_spent:,.2f}  ({remaining_budget_str})")

        unlabeled_available = unlabeled_pool

        loop_data_dir = os.path.join(output_dir, "loop_data")
        os.makedirs(loop_data_dir, exist_ok=True)

        df_clause.to_pickle(f"{loop_data_dir}/df_clause.pkl")
        with open(f"{loop_data_dir}/embedding_cache.pkl", 'wb') as f:
            pickle.dump(embedding_cache, f)

        if test and remaining_llm_calls is not None:
            remaining_llm_calls.to_pickle(f"{loop_data_dir}/remaining_llm_calls.pkl")
            remaining_llm_calls_path = f"{loop_data_dir}/remaining_llm_calls.pkl"
        else:
            remaining_llm_calls_path = None

        if not test and unlabeled_pairs is not None:
            unlabeled_pairs.to_pickle(f"{loop_data_dir}/unlabeled_pairs.pkl")
            unlabeled_pairs_path = f"{loop_data_dir}/unlabeled_pairs.pkl"
        else:
            unlabeled_pairs_path = None

        if isinstance(input_file, pd.DataFrame):
            input_csv = f"{loop_data_dir}/input_iter_{iteration}.csv"
            input_file.to_csv(input_csv, index=False)
            input_csv_path = input_csv
        else:
            input_csv_path = input_file

        output_notebook = os.path.join(output_dir, f"FEA_Pipeline_iter_{iteration}.ipynb")

        parameters = {
            'iteration_number': iteration,
            'input_csv_path': input_csv_path,
            'df_clause_path': f"{loop_data_dir}/df_clause.pkl",
            'embedding_cache_path': f"{loop_data_dir}/embedding_cache.pkl",
            'test': test,
            'remaining_llm_calls_path': remaining_llm_calls_path,
            'unlabeled_pairs_path': unlabeled_pairs_path,
            'sent_frac': sent_frac,
            'budget': budget,
        }

        print(f"Executing FEA_Pipeline.ipynb with input: {input_csv_path}")
        pm.execute_notebook(
            'FEA_Pipeline.ipynb',
            output_notebook,
            parameters=parameters,
        )

        output_csv = f"fea_iterations/llm_batch_iter_{iteration}.csv"

        if test:
            if os.path.exists(output_csv):
                df_new_pairs = pd.read_csv(output_csv)

                if len(df_new_pairs) == 0:
                    print(f"\n⚠ No new pairs in output this iteration — stopping.")
                    break

                if isinstance(input_file, str):
                    df_existing = pd.read_csv(input_file)
                else:
                    df_existing = input_file

                df_accumulated = pd.concat([df_existing, df_new_pairs], ignore_index=True)

                accumulated_csv = f"{loop_data_dir}/accumulated_labeled_iter_{iteration}.csv"
                df_accumulated.to_csv(accumulated_csv, index=False)

                input_file = accumulated_csv

                pairs_selected = len(df_new_pairs)
                print(f"✓ Output has {pairs_selected} NEW pairs")
                print(f"✓ Accumulated dataset now has {len(df_accumulated)} total labeled pairs")
            else:
                print(f"⚠ Warning: Expected output file not found: {output_csv}")
                break
        else:
            nb = sb.read_notebook(output_notebook)
            df_to_llm_retrieved = nb.scraps['df_to_llm'].data
            print(f"\n✓ Retrieved df_to_llm from scrapbook")
            print(f"  Shape: {df_to_llm_retrieved.shape}")
            print(f"  Columns: {list(df_to_llm_retrieved.columns)}")

            if len(df_to_llm_retrieved) == 0:
                print(f"\n⚠ No pairs selected for LLM this iteration (empty df_to_llm).")
                print(f"  This usually means too few entailed pairs — skipping LLM call.")
                break

            df_to_llm_file = f"{loop_data_dir}/df_to_llm_iter_{iteration}.csv"
            df_to_llm_retrieved.to_csv(df_to_llm_file, index=False)
            print(f"✓ Saved df_to_llm to {df_to_llm_file}")

            next_input_file = f"{labeled_pairs_dir}/Results_DS_BtoS_iteration_{iteration + 1}.csv"

            # --- Step 1: Send forward-direction pairs to LLM ---
            one_way_output = f"{labeled_pairs_dir}/Results_DS_BtoS_iteration_{iteration + 1}_one_way"

            evaluator_notebook = os.path.join(output_dir, f"Evaluator_iter_{iteration}.ipynb")
            evaluator_params = {
                'llm_model': "deepseek-reasoner",
                'input_file': df_to_llm_file,
                'args_file': "ArgLevel_ClauseIds_df.xlsx",
                'prompt': "test_prompt_tot_json2",
                'output': one_way_output,
                'previous_input_file': '',  # Don't merge with previous yet
            }

            print(f"\n{'='*60}")
            print(f"EXECUTING LLM EVALUATOR (forward direction)")
            print(f"{'='*60}")
            print(f"Sending {len(df_to_llm_retrieved)} pairs to LLM...")

            if deepseek_api_key:
                os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key

            pm.execute_notebook(
                'llm_calls/2_3_ExecuteEvaluator.ipynb',
                evaluator_notebook,
                parameters=evaluator_params,
            )
            print(f"✓ Forward LLM evaluation complete")

            # --- Step 2: Process one-way results bidirectionally ---
            one_way_csv = f"{one_way_output}.csv"
            if os.path.exists(one_way_csv):
                df_one_way = pd.read_csv(one_way_csv)
            else:
                print(f"⚠ ERROR: One-way output not found: {one_way_csv}")
                break

            df_bidirectional = process_llm_results_bidirectional(
                df_one_way=df_one_way,
                df_clause=df_clause,
                results_output_path=next_input_file,
                model="deepseek-reasoner",
                prompt_type="test_prompt_tot_json2",
                args_file="ArgLevel_ClauseIds_df.xlsx",
                output_dir=labeled_pairs_dir,
                batch_label=f"reverse_iter_{iteration}",
                deepseek_api_key=deepseek_api_key,
                max_reverse_pairs=100_000,
            )

            # --- Step 3: Merge with previous accumulated results ---
            if os.path.exists(input_csv_path):
                df_prev = pd.read_csv(input_csv_path)
                df_merged = pd.concat([df_prev, df_bidirectional], ignore_index=True)
                df_merged.to_csv(next_input_file, index=False)
                print(f"✓ Merged {len(df_prev)} previous + {len(df_bidirectional)} new = {len(df_merged)} total")
            else:
                df_bidirectional.to_csv(next_input_file, index=False)

            if os.path.exists(next_input_file):
                input_file = next_input_file
                pairs_selected = len(df_to_llm_retrieved)
                print(f"✓ Processed {pairs_selected} NEW pairs")
            else:
                print(f"⚠ ERROR: Expected output file not found: {next_input_file}")
                break

        df_current = pd.read_csv(output_csv) if test else df_to_llm_retrieved
        pairs_selected = len(df_current)

        cumulative_labeled += pairs_selected
        unlabeled_pool -= pairs_selected

        # Track cost (forward + any reverse pairs sent to LLM)
        iteration_cost = pairs_selected * cost_per_pair
        cost_spent += iteration_cost

        if cumulative_labeled + unlabeled_pool != total_pairs:
            print(f"⚠ WARNING: Total pairs mismatch! {cumulative_labeled} + {unlabeled_pool} != {total_pairs}")

        if test:
            stats = calculate_entailment_ratio(df_current)
            entailment_ratio = stats['entailment_ratio']
            print(f"    Bidirectionally entailed: {stats['bidirectionally_entailed']}/{stats['total_pairs']} ({entailment_ratio:.1%})")
        else:
            entailment_ratio = 0.0

        if test and remaining_llm_calls_path and os.path.exists(remaining_llm_calls_path):
            remaining_llm_calls = pd.read_pickle(remaining_llm_calls_path)
            actual_unlabeled = len(remaining_llm_calls)
            if actual_unlabeled != unlabeled_pool:
                print(f"⚠ WARNING: Unlabeled count mismatch! Expected {unlabeled_pool}, got {actual_unlabeled} in pickle")
                print(f"   This means FEA_Pipeline may not be properly updating remaining_llm_calls")
                unlabeled_pool = actual_unlabeled
        elif not test and unlabeled_pairs_path and os.path.exists(unlabeled_pairs_path):
            unlabeled_pairs = pd.read_pickle(unlabeled_pairs_path)
            actual_unlabeled = len(unlabeled_pairs)
            if actual_unlabeled != unlabeled_pool:
                print(f"⚠ WARNING: Unlabeled count mismatch! Expected {unlabeled_pool}, got {actual_unlabeled} in pickle")
                print(f"   This means FEA_Pipeline may not be properly updating unlabeled_pairs")
                unlabeled_pool = actual_unlabeled

        unlabeled_remaining = unlabeled_pool

        iteration_stats.append({
            'iteration': iteration,
            'labeled_input': cumulative_labeled - pairs_selected,
            'unlabeled_available': unlabeled_available,
            'pairs_selected': pairs_selected,
            'cumulative_labeled': cumulative_labeled,
            'unlabeled_remaining': unlabeled_remaining,
            'output_file': output_csv if test else next_input_file,
            'entailment_ratio': entailment_ratio,
            'iteration_cost': iteration_cost,
            'cumulative_cost': cost_spent,
        })

        print(f"  → Labeled input: {cumulative_labeled - pairs_selected}")
        print(f"  → Unlabeled available: {unlabeled_available}")
        print(f"  → Iteration cost: ${iteration_cost:,.2f}  (cumulative: ${cost_spent:,.2f})")

    print(f"\n{'='*60}")
    print("FEA Loop Complete!")
    print(f"{'='*60}")

    return iteration_stats


def save_iteration_stats(
    iteration_stats: List[dict],
    output_dir: str,
    cost_per_pair: float = 0.002,
) -> pd.DataFrame:
    """
    Print and save iteration statistics with cost summary.

    Parameters
    ----------
    iteration_stats : list[dict]
        List of per-iteration stat dictionaries (from :func:`run_fea_loop`).
    output_dir : str
        Directory in which to save ``iteration_stats.csv``.
    cost_per_pair : float
        Fallback cost per pair (used only if iteration dicts lack
        ``cumulative_cost``).

    Returns
    -------
    pd.DataFrame
        The statistics DataFrame (also saved to disk).
    """
    stats_df = pd.DataFrame(iteration_stats)
    print("\nIteration Statistics:")
    print(stats_df)

    total_pairs_sent = stats_df['pairs_selected'].sum()

    # Use tracked cost if available, else fall back to simple estimate
    if 'cumulative_cost' in stats_df.columns and len(stats_df) > 0:
        total_cost = stats_df['cumulative_cost'].iloc[-1]
    else:
        total_cost = total_pairs_sent * cost_per_pair

    print(f"\n{'='*60}")
    print("COST SUMMARY")
    print(f"{'='*60}")
    print(f"Total pairs sent to LLM: {total_pairs_sent:,}")
    print(f"Total cost: ${total_cost:,.2f}")
    print(f"{'='*60}")

    stats_df['total_cost'] = total_cost
    csv_path = os.path.join(output_dir, "iteration_stats.csv")
    stats_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved iteration statistics to {csv_path}")

    return stats_df

def estimate_thresholds(
    df_p: pd.DataFrame,
    df_sc: pd.DataFrame,
    n_trials: int = 20,
    n_pairs: int = 1000,
    percentile: float = 99.0,
    model_path: str = "./fine_tuned_bi_model",
    embedding_cache: Optional[Dict[str, np.ndarray]] = None,
    id_col: str = "sentence_id",
    text_col: str = "sentence",
    base_seed: int = 0,
) -> pd.DataFrame:
    """
    Estimate the stability of cosine-similarity thresholds for BB and BS
    pairs by repeating the sampling process ``n_trials`` times with
    different random seeds.

    Each trial:
    1. Generates ``n_pairs`` Book-Book and ``n_pairs`` Book-Speech pairs
       via :func:`generate_valid_pairs_by_type`.
    2. Computes cosine similarity with the fine-tuned bi-encoder
       via :func:`generate_new_bert_results` (using the embedding cache
       so embeddings are computed only once).
    3. Takes the ``percentile``-th percentile of each distribution as
       that trial's threshold.

    Parameters
    ----------
    df_p : pd.DataFrame
        Premise sentences.
    df_sc : pd.DataFrame
        Conclusion sentences.
    n_trials : int, default 20
        Number of independent random samples to draw.
    n_pairs : int, default 1000
        How many pairs per type per trial.
    percentile : float, default 99.0
        Percentile used as the "top-X %" threshold (99 → top 1 %).
    model_path : str
        Path to the fine-tuned bi-encoder model.
    embedding_cache : dict or None
        Pre-computed embeddings ``{id: ndarray}``.
    id_col, text_col : str
        Column names in df_p / df_sc.
    base_seed : int, default 0
        First seed; trial *i* uses ``base_seed + i``.

    Returns
    -------
    pd.DataFrame
        One row per trial with columns
        ``['trial', 'seed', 'threshold_bb', 'threshold_bs']``.
    """
    rows = []
    for i in range(n_trials):
        seed = base_seed + i
        print(f"  Trial {i+1}/{n_trials}  (seed={seed})", end=" … ")

        df_bb, df_bs = generate_valid_pairs_by_type(
            df_p, df_sc, n=n_pairs,
            id_col=id_col, text_col=text_col,
            random_seed=seed,
        )

        df_bb = generate_new_bert_results(
            df_bb,
            text_col1="text1", text_col2="text2",
            model_path=model_path,
            new_col="cosine_sim",
            embedding_cache=embedding_cache,
            id_col1="id1", id_col2="id2",
        )
        df_bs = generate_new_bert_results(
            df_bs,
            text_col1="text1", text_col2="text2",
            model_path=model_path,
            new_col="cosine_sim",
            embedding_cache=embedding_cache,
            id_col1="id1", id_col2="id2",
        )

        t_bb = float(np.percentile(df_bb["cosine_sim"].dropna(), percentile))
        t_bs = float(np.percentile(df_bs["cosine_sim"].dropna(), percentile))

        print(f"BB={t_bb:.6f}  BS={t_bs:.6f}")

        rows.append({
            "trial": i + 1,
            "seed": seed,
            "threshold_bb": t_bb,
            "threshold_bs": t_bs,
        })

    return pd.DataFrame(rows)
