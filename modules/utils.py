import numpy as np
import pandas as pd

def take_two_sides_extreme_sorted(
    df: pd.DataFrame, 
    n_extreme: int,
    part_column: str=None,
    head_value: str='',
    tail_value: str=''
) -> pd.DataFrame:

    head_df = df.head(n_extreme)[:]
    tail_df = df.tail(n_extreme)[:]

    if part_column is not None:
        head_df[part_column] = head_value
        tail_df[part_column] = tail_value

    return (pd.concat([head_df, tail_df])
            .drop_duplicates()
            .reset_index(drop=True))

def normalize(
    v: np.ndarray
) -> np.ndarray:

    """Normalize a 1-D vector."""
    if v.ndim != 1:
        raise ValueError('v should be 1-D, {}-D was given'.format(
            v.ndim))
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def project_params(
    u: np.ndarray, 
    v: np.ndarray
) -> np.ndarray:

    """Projecting and rejecting the vector v onto direction u with scalar."""
    normalize_u = normalize(u)
    projection = (v @ normalize_u)
    projected_vector = projection * normalize_u
    rejected_vector = v - projected_vector
    return projection, projected_vector, rejected_vector


def cosine_similarity(
    v: np.ndarray, 
    u: np.ndarray
) -> np.ndarray:

    """Calculate the cosine similarity between two vectors."""
    v_norm = np.linalg.norm(v)
    u_norm = np.linalg.norm(u)
    similarity = v @ u / (v_norm * u_norm)
    return similarity