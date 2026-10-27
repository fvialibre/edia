import numpy as np
import pandas as pd
import pytz, argparse
from datetime import datetime
from typing import List, Tuple, Dict, Any

def parse_cmd_line_args(
) -> Dict[str, Any]:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Exension')
    parser.add_argument('-p', '--port', type=int, default=7860, help='Config APP port(default: %(default)s)')
    return vars(parser.parse_args())

class DateLogs:
    def __init__(
        self,
        zone: str = "America/Argentina/Cordoba"
    ) -> None:

        self.time_zone = pytz.timezone(zone)

    def full(
        self
    ) -> str:

        now = datetime.now(self.time_zone)
        return now.strftime("%H:%M:%S %d-%m-%Y")

    def day(
        self
    ) -> str:

        now = datetime.now(self.time_zone)
        return now.strftime("%d-%m-%Y")

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
        raise ValueError(f'v should be 1-D, {v.ndim}-D was given')
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


def axes_labels_format(
    left: str, 
    right: str, 
    sep: str, 
    word_wrap: int = 4
) -> str:

    def sparse(
        word: str, 
        max_len: int
    ) -> str:

        diff = max_len-len(word)
        rest = diff if diff > 0 else 0
        return word+" "*rest

    def gen_block(
        list_: List[str], 
        n_rows:int, 
        n_cols:int
    ) -> List[str]:

        block = []
        block_row = []
        for r in range(n_rows):
            for c in range(n_cols):
                i = r * n_cols + c
                w = list_[i] if i <= len(list_) - 1 else ""
                block_row.append(w)
                if (i+1) % n_cols == 0:
                    block.append(block_row)
                    block_row = []
        return block

    # Transform 'string' to list of string
    l_list = [word.strip() for word in left.split(",") if word.strip() != ""]
    r_list = [word.strip() for word in right.split(",") if word.strip() != ""]
    
    # Get longest word, and longest_list
    longest_list = max(len(l_list), len(r_list))
    longest_word = len(max( max(l_list, key=len), max(r_list, key=len)))

    # Creation of word blocks for each list 
    n_rows =  (longest_list // word_wrap) if longest_list % word_wrap == 0 else (longest_list // word_wrap) + 1
    n_cols = word_wrap

    l_block = gen_block(l_list, n_rows, n_cols)
    r_block = gen_block(r_list, n_rows, n_cols)

    # Transform list of list to sparse string
    labels = ""
    for i,(l,r) in enumerate(zip(l_block, r_block)):
        line = ' '.join([sparse(w, longest_word) for w in l]) + sep + \
                ' '.join([sparse(w, longest_word) for w in r])
        labels += f"← {line} →\n" if i==0 else f"  {line}  \n"

    return labels

def sort_pllScores(
    pll_scores: Dict[str,int]
) -> List[Tuple[str, int]]:

    # Sorted pll scores in descending order
    return  sorted(pll_scores.items(), key=lambda x: x[1], reverse=True)
