import pandas as pd
import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer, util, InputExample, losses, CrossEncoder
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import List, Tuple, Any, Mapping, Iterable, Dict, Literal, Union, Optional
import plotly.graph_objects as go


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
def merge_pairwise_texts(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df1_cols: List[str],
    df2_cols: List[str],
) -> pd.DataFrame:
    """
    Merge a text dataframe (df1) with a pairwise dataframe (df2) to obtain
    (id1, id2, text1, text2, verdict), where 'verdict' is optional.

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

    # Prepare df1 for merging on id1 and id2
    df1_for_id1 = df1_reduced.rename(
        columns={id_col_df1: "id1", text_col_df1: "text1"}
    )
    df1_for_id2 = df1_reduced.rename(
        columns={id_col_df1: "id2", text_col_df1: "text2"}
    )

    # Rename id columns in df2 to standard names
    merged = df2_reduced.rename(
        columns={id1_col_df2: "id1", id2_col_df2: "id2"}
    )

    # Attach text1 (for id1)
    merged = merged.merge(df1_for_id1, on="id1", how="left")

    # Attach text2 (for id2)
    merged = merged.merge(df1_for_id2, on="id2", how="left")

    # Standardize/construct verdict column
    if verdict_col_df2 is not None:
        merged = merged.rename(columns={verdict_col_df2: "verdict"})
    else:
        merged["verdict"] = np.nan

    # Reorder columns
    return merged[["id1", "id2", "text1", "text2", "verdict"]]

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
    # Stack embeddings into 2D arrays (n_samples, dim)
    emb1 = np.stack(df[emb_col1].to_numpy())
    emb2 = np.stack(df[emb_col2].to_numpy())

    # Compute numerator: dot product row-wise
    numer = np.sum(emb1 * emb2, axis=1)

    # Compute norms
    norm1 = np.linalg.norm(emb1, axis=1)
    norm2 = np.linalg.norm(emb2, axis=1)

    # Avoid division by zero
    denom = norm1 * norm2
    # Use np.where to handle zero norms safely
    cos_sim = np.where(denom > 0, numer / denom, 0.0)

    df[new_col] = cos_sim
    return df

def add_cosine_similarity_from_text(
    df: pd.DataFrame,
    text_col1: str,
    text_col2: str,
    model_name: str = "BAAI/bge-en-icl",
    new_col: str = "cosine_sim",
    batch_size: int = 128,  
    show_progress_bar: bool = True
) -> pd.DataFrame:
    """
    Encode two text columns with a SentenceTransformer model and compute
    row-wise cosine similarity between them in a vectorized way.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing two text columns.
    text_col1 : str
        Name of the first text column.
    text_col2 : str
        Name of the second text column.
    model_name : str, default "BAAI/bge-en-icl"
        SentenceTransformer model to use.
    new_col : str, default "cosine_sim"
        Name of the new column with cosine similarities.
    batch_size : int, default 128
        Batch size for encoding.
    show_progress_bar : bool, default False
        Whether to show progress bar during encoding.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with an additional column `new_col` containing
        cosine similarities between the embeddings of the two text columns.
    """
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

    # Planned sizes
    n1 = int(round(frac1 * n))
    n2 = int(round(frac2 * n))

    if n1 + n2 > n:
        raise ValueError(
            f"Requested subsamples too large: n1 + n2 = {n1 + n2} > n = {n}. "
            "Reduce frac1 and/or frac2 so that frac1 + frac2 <= 1."
        )

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
    new_col: str = "bert_score"
) -> pd.DataFrame:
    """
    Runs inference using the newly fine-tuned model saved at `model_path`.
    This is essentially a wrapper around add_cosine_similarity_from_text 
    but points to the local folder.
    """
    # Reuse the optimized inference function defined earlier
    return add_cosine_similarity_from_text(
        df=df,
        text_col1=text_col1,
        text_col2=text_col2,
        model_name=model_path, # Load from local folder
        new_col=new_col,
        batch_size=128 # Inference batch size can be slightly larger than training
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
    transitivity_col: str = "transitivity_score"
) -> pd.DataFrame:
    """
    Uses the trained regression model to predict probability (0 to 1).
    
    Hard Constraint:
    If 'transitivity_col' is present, any row with transitivity_score == 1.0 
    will have its probability FORCED to 1.0, overriding the model.
    """
    df_out = df.copy()
    
    # Handle rows that might have NaN features by skipping them or filling
    mask_valid = df_out[feature_cols].notna().all(axis=1)
    
    if mask_valid.any():
        X = df_out.loc[mask_valid, feature_cols].values
        # predict_proba returns [prob_class_0, prob_class_1]
        probs = model_pipeline.predict_proba(X)[:, 1]
        df_out.loc[mask_valid, new_col] = probs
    
    # Default NaNs
    df_out.loc[~mask_valid, new_col] = np.nan
    
    # --- Transitivity Override ---
    # Disabled upon request: We no longer force probability to 1.0 even if transitivity is 1.0.
    # The model should learn to trust this feature if it's reliable.
    
    return df_out

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
            
            # We use cross_val_predict to generate "clean" predictions for every row
            # The model is trained on K-1 folds and predicts on the Kth fold.
            y_probs = cross_val_predict(
                pipeline, 
                X, 
                y_true, 
                cv=5, 
                method='predict_proba'
            )[:, 1]

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
        
        # 3-Fold Cross-Validation for robustness
        scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc')
        return scores.mean()

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


def calculate_llm_savings_stats(
    df: pd.DataFrame,
    prob_col: str,
    verdict_col: str,
    threshold: float,
    positive_label: str = "YES",
    n_sample: int = 10000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Calculates statistics for the "Send to LLM" decision logic.
    
    Logic:
    1. Probability <= Threshold -> Auto-Reject (Predicted Not Entailed).
       (The model says 'No' or 'Low Confidence', and we trust it to be No)
    2. Probability > Threshold -> Send to LLM (Candidate for Entailment).
       (The model says 'Maybe Yes', so we verify with LLM)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    prob_col : str
        Probability column.
    verdict_col : str
        Ground truth column.
    threshold : float
        Threshold (tau).
    positive_label : str, default "YES"
        Label for positive class.
    n_sample : int, default 10000
        Sample size.
    random_state : int, default 42
        Random seed.

    Returns
    -------
    dict
        Dictionary of counts and stats.
    """
    # 1. Sample
    if len(df) <= n_sample:
        # print(f"Dataset size ({len(df)}) is smaller than requested sample ({n_sample}). Using full dataset.")
        df_sub = df.copy()
    else:
        df_sub = df.sample(n=n_sample, random_state=random_state).copy()
    
    # 2. Logic
    actual_pos = df_sub[verdict_col] == positive_label
    actual_neg = ~actual_pos
    probs = df_sub[prob_col]

    # Zones
    mask_sent = probs > threshold
    mask_auto_reject = ~mask_sent # probs <= threshold

    # Counts
    # Sent to LLM (Model said > Tau)
    Sent_Yes = (mask_sent & actual_pos).sum()      # Costly but Necessary (Verified Positive)
    Sent_No  = (mask_sent & actual_neg).sum()      # Costly Validated (Verified Negative)
    
    # Auto-Reject (Model said <= Tau)
    TN_rej = (mask_auto_reject & actual_neg).sum() # Good (Efficiency)
    FN_rej = (mask_auto_reject & actual_pos).sum() # Bad  (Lost Opportunity)
    
    total = len(df_sub)
    total_sent = Sent_Yes + Sent_No
    
    stats = {
        "TN_AutoReject": int(TN_rej),
        "FN_AutoReject": int(FN_rej),
        "Sent_Yes": int(Sent_Yes),
        "Sent_No": int(Sent_No),
        "Total": int(total),
        "Sent_Total": int(total_sent),
        "Threshold": threshold
    }
    
    # display report
    print(f"\\n>>> Savings Analysis (Sample n={total})")
    print(f"Threshold: Tau={threshold:.4f} (Send if > Tau)")
    print("-" * 65)
    print(f"{'Metric':<35} | {'Count':<10} | {'Rate':<10}")
    print("-" * 65)
    print(f"{'Auto-Reject (Correct)':<35} | {TN_rej:<10} | {TN_rej/total:.2%}")
    print(f"{'Auto-Reject (Wrong - FN)':<35} | {FN_rej:<10} | {FN_rej/total:.2%}")
    print(f"{'Sent to LLM (Actually Yes)':<35} | {Sent_Yes:<10} | {Sent_Yes/total:.2%}")
    print(f"{'Sent to LLM (Actually No)':<35} | {Sent_No:<10} | {Sent_No/total:.2%}")
    print("-" * 65)
    
    # LLM Workload Reduction
    reduction = (total - total_sent) / total
    print(f"LLM Calls: {total_sent} ({total_sent/total:.2%})")
    print(f"LLM Savings: {total - total_sent} ({reduction:.2%})")
    
    return stats


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
    keep_cols: List[str] = ['id1', 'id2', 'text1', 'text2', 'entailment_probability']
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
    
    # Handle renaming only if prob_col is different but 'entailment_probability' is requested
    df_temp = df.copy()
    if prob_col != 'entailment_probability' and 'entailment_probability' in keep_cols:
        df_temp = df_temp.rename(columns={prob_col: 'entailment_probability'})
    
    # Ensure columns exist before selecting
    missing = [c for c in keep_cols if c not in df_temp.columns]
    if missing:
        raise ValueError(f"Missing required columns for final output: {missing}")
        
    df_out = df_temp.loc[mask, keep_cols].copy()
    
    print(f"--- Generating LLM Batch ---")
    print(f"Original Count: {len(df):,}")
    print(f"Filtered Count: {len(df_out):,} ({(len(df_out)/len(df)):.1%})")
    print(f"Condition:      P > {threshold:.4f} (Send High Confidence Pairs)")
    
    return df_out






