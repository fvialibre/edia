import random

import pandas as pd


def _add_validation_count(df, df_validations, identity_col, attribute_col):
    """
    Adds a `validation_count` column to `df` by counting occurrences of
    (identity, attribute) pairs in `df_validations`.

    Args:
        df: DataFrame containing data points (workshop or SeeGULL).
        df_validations: DataFrame with validation records.
        identity_col: Name of the identity column in `df`.
        attribute_col: Name of the attribute column in `df`.

    Returns:
        DataFrame: `df` with an added `validation_count` column.
    """
    # Count validations for each (identity, attribute) pair in df_validations
    validation_counts = (
        df_validations.groupby(["identity", "attribute"])
        .size()
        .reset_index(name="validation_count")
    )

    # Rename columns in validation_counts to match df’s column names
    validation_counts.columns = [identity_col, attribute_col, "validation_count"]

    # Merge on the matching column names
    df_with_counts = df.merge(
        validation_counts, on=[identity_col, attribute_col], how="left"
    )

    # Fill NaNs
    df_with_counts["validation_count"] = (
        df_with_counts["validation_count"].fillna(0).astype(int)
    )

    return df_with_counts


def _get_excluded_pairs(
    df_ws_validations, df_ws_stereotypes, annotator_id, df_skips=None, skip_threshold=3
):
    """
    Compute the set of (identity, attribute) pairs excluded for the annotator.
    Now also excludes pairs that have been skipped more than the threshold number of times.

    Args:
        df_ws_validations: DataFrame with validation records.
        df_ws_stereotypes: DataFrame with stereotypes created by annotators.
        annotator_id: ID of the current annotator.
        df_skips: DataFrame with skip counts.
        skip_threshold: Minimum number of skips to exclude a pair.
    """
    result = set()

    # Collect pairs from validations and stereotypes (if annotator_id is provided)
    if annotator_id is not None:
        # Collect pairs from validations
        validation_pairs = set(
            df_ws_validations[df_ws_validations["annotator_id"] == annotator_id][
                ["identity", "attribute"]
            ].itertuples(index=False, name=None)
        )
        # Collect pairs from created stereotypes
        stereotype_pairs = set(
            df_ws_stereotypes[df_ws_stereotypes["annotator_id"] == annotator_id][
                ["identity", "attribute"]
            ].itertuples(index=False, name=None)
        )
        result = validation_pairs.union(stereotype_pairs)

    # Add pairs skipped by this annotator
    if annotator_id is not None and df_skips is not None and not df_skips.empty:
        annotator_skipped_pairs = set(
            df_skips[df_skips["annotator_id"] == annotator_id][
                ["identity", "attribute"]
            ].itertuples(index=False, name=None)
        )
        result = result.union(annotator_skipped_pairs)

    # Add globally frequently skipped pairs to exclusions
    if df_skips is not None and not df_skips.empty:
        # Calculate global skip counts dynamically
        skip_counts = (
            df_skips.groupby(["identity", "attribute"])
            .size()
            .reset_index(name="skip_count")
        )
        # Filter for frequently skipped pairs
        frequent_skips = skip_counts[skip_counts["skip_count"]
                                     >= skip_threshold]
        # Extract pairs
        skipped_pairs = set(
            frequent_skips[["identity", "attribute"]].itertuples(
                index=False, name=None
            )
        )
        result = result.union(skipped_pairs)

    return result


def _filter_candidates(
    df, validation_col, identity_col, attribute_col, excluded_pairs, max_validation
):
    """Filter candidates based on validation count and exclusions."""
    filtered = df[df[validation_col] < max_validation]
    if excluded_pairs:
        filtered = filtered[
            ~filtered.apply(
                lambda r: (r[identity_col], r[attribute_col]) in excluded_pairs, axis=1
            )
        ]
    return filtered


def _select_candidate(df, validation_col):
    """Select a candidate with the minimum validation count from the filtered dataframe."""
    if df.empty:
        return None
    min_validation = df[validation_col].min()
    candidates = df[df[validation_col] == min_validation]
    return candidates.sample(1).iloc[[0]]


def _get_fallback_candidate(df_seegull, excluded_pairs):
    """Select a fallback candidate from SeeGULL with the global minimum validation count."""
    global_min = df_seegull["validation_count"].min()
    candidates = df_seegull[df_seegull["validation_count"] == global_min]
    # Attempt to exclude pairs if possible
    if excluded_pairs:
        filtered = candidates[
            ~candidates.apply(
                lambda r: (r["identity_country_name"], r["translated_attribute_list"])
                in excluded_pairs,
                axis=1,
            )
        ]
        if not filtered.empty:
            return filtered.sample(1).iloc[0]
    # Return any candidate if all are excluded or no exclusions
    return candidates.sample(1).iloc[0]


def _get_fallback_candidate_heseia(df_heseia, excluded_pairs):
    """Select a fallback candidate from HESEIA with the global minimum validation count."""
    global_min = df_heseia["validation_count"].min()
    candidates = df_heseia[df_heseia["validation_count"] == global_min]
    # Attempt to exclude pairs if possible
    if excluded_pairs:
        filtered = candidates[
            ~filtered.apply(
                lambda r: (r["region"], r["attribute"]) in excluded_pairs,
                axis=1,
            )
        ]
        if not filtered.empty:
            return filtered.sample(1).iloc[0]
    # Return any candidate if all are excluded or no exclusions
    return candidates.sample(1).iloc[0]


def _get_neighbor_set(df_borders, user_nationalities):
    """
    Return a set of neighboring countries for all user_nationalities
    based on df_borders columns: 'country_name' and 'country_border_name'.
    """
    neighbors = set()
    if df_borders is not None and not df_borders.empty and user_nationalities:
        for row in df_borders.itertuples(index=False):
            if row.country_name in user_nationalities:
                neighbors.add(row.country_border_name)
    return neighbors


def _pick_same_neighbor_other(
    df,
    col_nationalities,
    user_nationalities,
    df_borders,
    p_neighbor,
    p_other,
    excluded_pairs,
    validation_col,
    identity_col,
    attribute_col,
    max_validation=3,
):
    """
    For a given dataset (workshop or SeeGULL), pick a row from:
    - same nationality set,
    - neighboring nationality set, or
    - other set,
    with probabilities p_same = 1 - p_neighbor - p_other, p_neighbor, p_other.
    If any set is empty, it is removed and probabilities are renormalized accordingly.
    Returns a candidate row or None if nothing found.
    """
    if not user_nationalities:
        return None

    neighbor_countries = _get_neighbor_set(df_borders, user_nationalities)

    same_df = df.loc[
        df[col_nationalities].apply(
            lambda x: (
                any(c in x for c in user_nationalities)
                if isinstance(x, list)
                else False
            )
        ),
        :,
    ]
    same_df = _filter_candidates(
        same_df,
        validation_col,
        identity_col,
        attribute_col,
        excluded_pairs,
        max_validation,
    )

    neighbor_df = df.loc[
        df[col_nationalities].apply(
            lambda x: (
                any(c in neighbor_countries for c in x)
                if isinstance(x, list)
                else False
            )
        ),
        :,
    ]
    neighbor_df = _filter_candidates(
        neighbor_df,
        validation_col,
        identity_col,
        attribute_col,
        excluded_pairs,
        max_validation,
    )

    def not_same_or_neighbor(x):
        if not isinstance(x, list):
            return False
        return not any(c in user_nationalities for c in x) and not any(
            c in neighbor_countries for c in x
        )

    other_df = df.loc[df[col_nationalities].apply(not_same_or_neighbor), :]
    other_df = _filter_candidates(
        other_df,
        validation_col,
        identity_col,
        attribute_col,
        excluded_pairs,
        max_validation,
    )

    p_dist = {
        "same": max(0.0, 1.0 - p_neighbor - p_other),
        "neighbor": p_neighbor,
        "other": p_other,
    }
    sets_map = {"same": same_df, "neighbor": neighbor_df, "other": other_df}
    valid_keys = ["same", "neighbor", "other"]

    while valid_keys:
        total_p = sum(p_dist[k] for k in valid_keys)
        if total_p <= 1e-12:
            return None
        r = random.random() * total_p
        cum = 0.0
        chosen = None
        for k in valid_keys:
            cum += p_dist[k]
            if r <= cum:
                chosen = k
                break

        chosen_df = sets_map[chosen]
        candidate = _select_candidate(chosen_df, validation_col)
        if candidate is not None:
            return candidate

        valid_keys.remove(chosen)
        if valid_keys:
            removed_prob = p_dist[chosen]
            for k in valid_keys:
                if total_p - removed_prob > 0.0:
                    p_dist[k] = p_dist[k] * (1.0 / (total_p - removed_prob))
                else:
                    p_dist[k] = 0.0

    return None


def _filter_about_nationality(
    df,
    identity_col,
    attribute_col,
    annotator_nationalities,
    validation_col,
    excluded_pairs,
    max_validation,
):
    """
    Filter dataframe to find rows where the identity column value is in annotator_nationalities.

    Args:
        df: DataFrame containing data points.
        identity_col: Name of the identity column in df.
        attribute_col: Name of the attribute column in df.
        annotator_nationalities: List of nationalities of the annotator.
        validation_col: Name of the validation count column.
        excluded_pairs: Set of (identity, attribute) pairs to exclude.
        max_validation: Maximum validation count to include.

    Returns:
        DataFrame: Filtered DataFrame.
    """
    if not annotator_nationalities:
        return pd.DataFrame()  # Return empty dataframe if no nationalities

    # Filter rows where the identity is in annotator_nationalities
    about_df = df[df[identity_col].isin(annotator_nationalities)]

    # Apply validation count and excluded pairs filters
    return _filter_candidates(
        about_df,
        validation_col,
        identity_col,
        attribute_col,
        excluded_pairs,
        max_validation,
    )


def select_data_point(
    df_ws_stereotypes,
    df_ws_validations,
    df_borders,
    df_heseia,
    df_seegull=None,
    df_skips=None,  # New parameter for skip counts
    annotator_id=None,
    annotator_nationalities=None,
    p_neighbor=0.2,
    p_other=0.1,
    p_step2_chance=0.4,  # Chance to execute step 2
    debug=False,
    skip_threshold=3,  # New parameter for skip threshold
) -> tuple: # Returns (identity, attribute, language_code)
    """
    Selects a random data point for validation based on a priority hierarchy that incorporates
    annotator nationality, validation count limits, and exclusion of previously annotated pairs
    and frequently skipped pairs.

    Selection Logic:
    1. Workshop Data (Same/Near/Other):
       - Prioritizes data points from the workshop dataset (`df_ws_stereotypes`). By default, a
         chance (1 - p_neighbor - p_other) for same nationality, p_neighbor chance for a neighboring
         country, p_other chance for a different country.
         If a chosen subset is empty, it is removed and probabilities are renormalized.
       - Excludes pairs the annotator has already validated or created (`excluded_pairs`).
       - Only selects data points with fewer than 3 validations (`validation_count < 3`).

    2. Workshop Data (About Annotator's Nationality):
       - With probability `p_step2_chance`, tries to find a stereotype where the identity is one
         of the annotator's nationalities. If this step is skipped (probability 1 - `p_step2_chance`),
         annotator nationalities are not provided or if no suitable candidate is found,
         proceeds to the next step.
       - Still enforces the validation count limit and exclusion of previously validated pairs.

    3. Workshop Data (Any Nationality):
       - If no valid candidates exist at the previous workshop stages, checks the entire workshop dataset
         for data points (no same/neighbor probability adjustments), with fewer than 3 validations.

    4. HESEIA Data:
       - If no workshop candidates are found, tries the HESEIA dataset, filtering only by validation count.
       - Only selects data points with fewer than 3 validations.

    5. SeeGULL Data (Same/Near/Other) [If SeeGULL is provided]:
       - If no HESEIA candidates are found, tries the SeeGULL dataset with similar logic
         as step 1: same country, neighbor, or different, with fewer than 3 validations.

    6. SeeGULL Data (About Annotator's Nationality) [If SeeGULL is provided]:
       - Tries to find a SeeGULL stereotype where the identity is one of the annotator's nationalities.
       - Still enforces the validation count limit and exclusion of previously validated pairs.

    7. SeeGULL Data (Any Nationality) [If SeeGULL is provided]:
       - If no same-neighbor-other SeeGULL candidates exist, checks SeeGULL for data points
         from any nationality (no probability adjustments).

    8. Fallback:
       - If all prior steps yield no candidates, selects a data point with the global
         minimum validation count from HESEIA.
       - Prioritizes excluded pair avoidance if possible.

    Parameters:
    -----------
    df_ws_stereotypes : pd.DataFrame
        Workshop-generated stereotypes with columns:
        - `identity`: The identity part of the stereotype.
        - `attribute`: The attribute part of the stereotype.
        - `annotator_id`: The ID of the annotator who created the stereotype.
        - `annotator_nationalities`: The nationalities of the annotator who created the stereotype.

    df_ws_validations : pd.DataFrame
        Workshop validation records with columns:
        - `identity`: The identity being validated.
        - `attribute`: The attribute being validated.
        - `annotator_id`: The ID of the annotator who performed the validation.

    df_borders : pd.DataFrame
        Dataframe with columns:
        - `country_name`: Name of a country.
        - `country_border_name`: Name of a country that borders `country_name`.

    df_heseia : pd.DataFrame
        HESEIA dataset with columns:
        - `region`: The identity part of the stereotype (corresponds to identity).
        - `attribute`: The attribute part of the stereotype.
        - `source_language`: The language code ('en', 'es', 'pt') of the source file.

    df_seegull : pd.DataFrame, optional
        Data from seegull_countries.csv with columns:
        - `identity_country_name`: The identity part of the stereotype.
        - `translated_attribute_list`: The attribute part of the stereotype.
        - `source_country`: The nationality of the annotator who created the stereotype.
        If not provided, SeeGULL-specific steps will be skipped.

    df_skips : pd.DataFrame, optional
        Dataframe with skip counts, containing columns:
        - `identity`: The identity part of the skipped stereotype.
        - `attribute`: The attribute part of the skipped stereotype.
        - `skip_count`: The number of times this pair has been skipped.

    skip_threshold : int, optional
        Minimum number of skips required to exclude a pair. Default 3.

    annotator_id : str, optional
        ID of the current annotator (for exclusion checks). Defaults to None.

    annotator_nationalities : list[str], optional
        List of nationalities for the annotator used for priority filtering.

    p_neighbor : float, optional
        Probability of picking a neighboring country. Default 0.2

    p_other : float, optional
        Probability of picking a different country. Default 0.1

    p_step2_chance : float, optional
        Probability of executing step 2 (Workshop: About Annotator's Nationality). Default 0.5

    debug : bool, optional
        If True, returns additional selection information. Default False

    Returns:
    --------
    tuple
        If debug=False: A tuple `(identity, attribute, language_code)` representing the selected data point and its source language.
        If debug=True: A tuple `(identity, attribute, language_code, selection_info)` where:
        - `identity`: The identity part of the stereotype
        - `attribute`: The attribute part of the stereotype
        - `language_code`: The source language code (e.g., 'en', 'es') or 'en' if unknown/not applicable.
        - `selection_info`: A dictionary with information about the selection process:
            - `source`: One of ["WS_SAME_NEAR_OTHER", "WS_ABOUT_NATIONALITY", "WS_ANY",
                              "HESEIA", "SG_SAME_NEAR_OTHER", "SG_ABOUT_NATIONALITY",
                              "SG_ANY", "FALLBACK_HESEIA"]
            - `category`: For SAME_NEAR_OTHER sources, one of ["same", "neighbor", "other"], otherwise None
            - `validation_count`: The number of validations for this data point before selection
    """
    excluded_pairs = _get_excluded_pairs(
        df_ws_validations, df_ws_stereotypes, annotator_id, df_skips, skip_threshold
    )

    # Add validation counts dynamically
    df_ws_stereotypes = _add_validation_count(
        df_ws_stereotypes, df_ws_validations, "identity", "attribute"
    )

    # Process SeeGULL if provided
    if df_seegull is not None:
        df_seegull["source_countries"] = df_seegull["source_country"].apply(
            lambda x: [x] if pd.notnull(x) else []
        )
        df_seegull = _add_validation_count(
            df_seegull,
            df_ws_validations,
            "identity_country_name",
            "translated_attribute_list",
        )

    # Add validation counts for HESEIA
    # Ensure df_heseia has the 'source_language' column before this step
    # This column should be added when df_heseia is created/loaded in interface_validator.py
    if 'source_language' not in df_heseia.columns:
         print("Warning: 'source_language' column missing in df_heseia. Defaulting language to 'en'.")
         # Add a default column if missing, although ideally it should be present
         df_heseia['source_language'] = 'en'

    df_heseia = _add_validation_count(
        df_heseia, df_ws_validations, "region", "attribute"
    )


    # Modified _pick_same_neighbor_other to return category information
    def _pick_with_category_info(
        df,
        col_nationalities,
        user_nationalities,
        df_borders,
        p_neighbor,
        p_other,
        excluded_pairs,
        validation_col,
        identity_col,
        attribute_col,
        max_validation=3,
    ):
        """Modified version that returns the selected category along with the candidate"""
        if not user_nationalities:
            return None, None

        neighbor_countries = _get_neighbor_set(df_borders, user_nationalities)

        same_df = df.loc[
            df[col_nationalities].apply(
                lambda x: (
                    any(c in x for c in user_nationalities)
                    if isinstance(x, list)
                    else False
                )
            ),
            :,
        ]
        same_df = _filter_candidates(
            same_df,
            validation_col,
            identity_col,
            attribute_col,
            excluded_pairs,
            max_validation,
        )

        neighbor_df = df.loc[
            df[col_nationalities].apply(
                lambda x: (
                    any(c in neighbor_countries for c in x)
                    if isinstance(x, list)
                    else False
                )
            ),
            :,
        ]
        neighbor_df = _filter_candidates(
            neighbor_df,
            validation_col,
            identity_col,
            attribute_col,
            excluded_pairs,
            max_validation,
        )

        def not_same_or_neighbor(x):
            if not isinstance(x, list):
                return False
            return not any(c in user_nationalities for c in x) and not any(
                c in neighbor_countries for c in x
            )

        other_df = df.loc[df[col_nationalities].apply(not_same_or_neighbor), :]
        other_df = _filter_candidates(
            other_df,
            validation_col,
            identity_col,
            attribute_col,
            excluded_pairs,
            max_validation,
        )

        p_dist = {
            "same": max(0.0, 1.0 - p_neighbor - p_other),
            "neighbor": p_neighbor,
            "other": p_other,
        }
        sets_map = {"same": same_df, "neighbor": neighbor_df, "other": other_df}
        valid_keys = ["same", "neighbor", "other"]

        while valid_keys:
            total_p = sum(p_dist[k] for k in valid_keys)
            if total_p <= 1e-12:
                return None, None
            r = random.random() * total_p
            cum = 0.0
            chosen = None
            for k in valid_keys:
                cum += p_dist[k]
                if r <= cum:
                    chosen = k
                    break

            chosen_df = sets_map[chosen]
            candidate = _select_candidate(chosen_df, validation_col)
            if candidate is not None:
                return candidate, chosen

            valid_keys.remove(chosen)
            if valid_keys:
                removed_prob = p_dist[chosen]
                for k in valid_keys:
                    if total_p - removed_prob > 0.0:
                        p_dist[k] = p_dist[k] * (1.0 / (total_p - removed_prob))
                    else:
                        p_dist[k] = 0.0

        return None, None

    # --- Selection Steps ---

    # 1) Workshop: same/neighbor/other
    if annotator_nationalities:
        candidate, category = _pick_with_category_info(
            df_ws_stereotypes,
            "annotator_nationalities",
            annotator_nationalities,
            df_borders,
            p_neighbor,
            p_other,
            excluded_pairs,
            "validation_count",
            "identity",
            "attribute",
            3,
        )
        if candidate is not None:
            selection_info = {
                "source": "WS_SAME_NEAR_OTHER",
                "category": category,
                "validation_count": candidate["validation_count"].iloc[0],
            }
            lang_code = 'en' # Workshop data is considered 'en' for now
            if debug:
                return (
                    candidate["identity"].iloc[0],
                    candidate["attribute"].iloc[0],
                    lang_code,
                    selection_info,
                )
            else:
                return (candidate["identity"].iloc[0], candidate["attribute"].iloc[0], lang_code)

    # 2) Workshop: about annotator's nationality (with probability p_step2_chance)
    if annotator_nationalities and random.random() < p_step2_chance:
        ws_about_nat = _filter_about_nationality(
            df_ws_stereotypes,
            "identity",
            "attribute",
            annotator_nationalities,
            "validation_count",
            excluded_pairs,
            3,
        )
        candidate = _select_candidate(ws_about_nat, "validation_count")
        if candidate is not None:
            selection_info = {
                "source": "WS_ABOUT_NATIONALITY",
                "category": None,
                "validation_count": candidate["validation_count"].iloc[0],
            }
            lang_code = 'en' # Workshop data is considered 'en' for now
            if debug:
                return (
                    candidate["identity"].iloc[0],
                    candidate["attribute"].iloc[0],
                    lang_code,
                    selection_info,
                )
            else:
                return (candidate["identity"].iloc[0], candidate["attribute"].iloc[0], lang_code)

    # 3) Workshop any nationality
    ws_filtered_any = _filter_candidates(
        df_ws_stereotypes,
        "validation_count",
        "identity",
        "attribute",
        excluded_pairs,
        3,
    )
    candidate = _select_candidate(ws_filtered_any, "validation_count")
    if candidate is not None:
        selection_info = {
            "source": "WS_ANY",
            "category": None,
            "validation_count": candidate["validation_count"].iloc[0],
        }
        lang_code = 'en' # Workshop data is considered 'en' for now
        if debug:
            return (
                    candidate["identity"].iloc[0],
                    candidate["attribute"].iloc[0],
                    lang_code,
                    selection_info,
                )
        else:
            return (candidate["identity"].iloc[0], candidate["attribute"].iloc[0], lang_code)

    # 4) HESEIA dataset
    heseia_filtered = _filter_candidates(
        df_heseia,
        "validation_count",
        "region",
        "attribute",
        excluded_pairs,
        3,
    )
    candidate = _select_candidate(heseia_filtered, "validation_count")
    if candidate is not None:
        selection_info = {
            "source": "HESEIA",
            "category": None,
            "validation_count": candidate["validation_count"].iloc[0],
        }
        # Extract source language, default to 'en' if column missing
        lang_code = candidate["source_language"].iloc[0] if "source_language" in candidate.columns else 'en'
        if debug:
            return (
                    candidate["region"].iloc[0],
                    candidate["attribute"].iloc[0],
                    lang_code,
                    selection_info,
                )
        else:
            return (candidate["region"].iloc[0], candidate["attribute"].iloc[0], lang_code)

    # 5) SeeGULL: same/neighbor/other (if available)
    if df_seegull is not None and annotator_nationalities:
        candidate, category = _pick_with_category_info(
            df_seegull,
            "source_countries",
            annotator_nationalities,
            df_borders,
            p_neighbor,
            p_other,
            excluded_pairs,
            "validation_count",
            "identity_country_name",
            "translated_attribute_list",
            3,
        )
        if candidate is not None:
            selection_info = {
                "source": "SG_SAME_NEAR_OTHER",
                "category": category,
                "validation_count": candidate["validation_count"].iloc[0],
            }
            lang_code = 'en' # Default language for SeeGULL data
            if debug:
                return (
                    candidate["identity_country_name"].iloc[0],
                    candidate["translated_attribute_list"].iloc[0],
                    lang_code,
                    selection_info,
                )
            else:
                return (
                    candidate["identity_country_name"].iloc[0],
                    candidate["translated_attribute_list"].iloc[0],
                    lang_code,
                )

    # 6) SeeGULL: about annotator's nationality (if available)
    if df_seegull is not None and annotator_nationalities:
        sg_about_nat = _filter_about_nationality(
            df_seegull,
            "identity_country_name",
            "translated_attribute_list",
            annotator_nationalities,
            "validation_count",
            excluded_pairs,
            3,
        )
        candidate = _select_candidate(sg_about_nat, "validation_count")
        if candidate is not None:
            selection_info = {
                "source": "SG_ABOUT_NATIONALITY",
                "category": None,
                "validation_count": candidate["validation_count"].iloc[0],
            }
            lang_code = 'en' # Default language for SeeGULL data
            if debug:
                return (
                    candidate["identity_country_name"].iloc[0],
                    candidate["translated_attribute_list"].iloc[0],
                    lang_code,
                    selection_info,
                )
            else:
                return (
                    candidate["identity_country_name"].iloc[0],
                    candidate["translated_attribute_list"].iloc[0],
                    lang_code,
                )

    # 7) SeeGULL any nationality (if available)
    if df_seegull is not None:
        sg_filtered_any = _filter_candidates(
            df_seegull,
            "validation_count",
            "identity_country_name",
            "translated_attribute_list",
            excluded_pairs,
            3,
        )
        candidate = _select_candidate(sg_filtered_any, "validation_count")
        if candidate is not None:
            selection_info = {
                "source": "SG_ANY",
                "category": None,
                "validation_count": candidate["validation_count"].iloc[0],
            }
            lang_code = 'en' # Default language for SeeGULL data
            if debug:
                return (
                    candidate["identity_country_name"].iloc[0],
                    candidate["translated_attribute_list"].iloc[0],
                    lang_code,
                    selection_info,
                )
            else:
                return (
                    candidate["identity_country_name"].iloc[0],
                    candidate["translated_attribute_list"].iloc[0],
                    lang_code,
                )

    # 8) Fallback (always use HESEIA)
    fallback_row = _get_fallback_candidate_heseia(df_heseia, excluded_pairs)
    selection_info = {
        "source": "FALLBACK_HESEIA",
        "category": None,
        "validation_count": fallback_row["validation_count"],
    }
    # Extract source language from fallback, default to 'en'
    lang_code = fallback_row["source_language"] if "source_language" in fallback_row else 'en'
    if debug:
        return (
            fallback_row["region"],
            fallback_row["attribute"],
            lang_code,
            selection_info,
        )
    else:
        return (
            fallback_row["region"],
            fallback_row["attribute"],
            lang_code,
        )
