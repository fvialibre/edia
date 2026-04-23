import json
import os
from datetime import datetime

import gradio as gr
from gradio_i18n import gettext as i18n
import pandas as pd

from interfaces.data_selection import select_data_point
from interfaces.nationalities import (
    nationalities,
    nationalities_es,
    nationalities_pt,
    translated_nationalities,
)

# --- Module-level constants ---

_NATIONALITY_LISTS = {
    "en": nationalities,
    "es": nationalities_es,
    "pt": nationalities_pt,
}

_COLOR_MAP = {"nationality": "red", "attribute": "green"}


def _get_nationality_choices(lang_code):
    nat_list = _NATIONALITY_LISTS.get(lang_code, nationalities)
    return list(zip(nat_list, nationalities))


def _get_divisions(countries):
    if not countries:
        return []
    if isinstance(countries, str):
        countries = [countries]
    try:
        df = pd.read_json("data/global_administrative_division.json")
        filtered = df[df["name"].isin(countries)]
        return sorted({
            f"{d['name']} ({row['name']})"
            for _, row in filtered.iterrows()
            for d in row["AD"]
        })
    except Exception:
        return []


# --- Interface ---

def interface(lang, token_id, age, gender, nationality, region, school, consent_checkbox):

    # --- One-time setup ---
    df_borders = pd.read_csv("data/country_borders.csv")
    os.makedirs("logs", exist_ok=True)
    for lc in _NATIONALITY_LISTS:
        for path, cols in [
            (f"logs/ws_stereotypes_{lc}.csv", ["identity", "attribute", "annotator_id", "annotator_nationalities", "source_language"]),
            (f"logs/ws_validations_{lc}.csv", ["identity", "attribute", "annotator_id"]),
            (f"logs/skips_{lc}.csv", ["identity", "attribute", "annotator_id"]),
        ]:
            if not os.path.exists(path):
                pd.DataFrame(columns=cols).to_csv(path, index=False)

    # --- Internal helpers ---

    def _load_csv(path, cols):
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            return pd.DataFrame(columns=cols)

    def _get_data_point(lang_code, token_id_val=None, nationality_val=None):
        if isinstance(nationality_val, str):
            nationality_val = [nationality_val] if nationality_val else None
        try:
            df_heseia = pd.read_csv(f"data/heseia_{lang_code}.csv")
            df_heseia["source_language"] = lang_code
        except Exception:
            df_heseia = pd.read_csv("data/heseia_en.csv")
            df_heseia["source_language"] = "en"
            lang_code = "en"

        df_stereotypes = _load_csv(
            f"logs/ws_stereotypes_{lang_code}.csv",
            ["identity", "attribute", "annotator_id", "annotator_nationalities", "source_language"],
        )
        df_validations = _load_csv(
            f"logs/ws_validations_{lang_code}.csv",
            ["identity", "attribute", "annotator_id"],
        )
        df_skips = _load_csv(f"logs/skips_{lang_code}.csv", ["identity", "attribute", "annotator_id"])

        return select_data_point(
            df_ws_stereotypes=df_stereotypes,
            df_ws_validations=df_validations,
            df_borders=df_borders,
            df_heseia=df_heseia,
            df_seegull=None,
            df_skips=df_skips if not df_skips.empty else None,
            annotator_id=token_id_val,
            annotator_nationalities=nationality_val,
        )

    def _log_skip(
        identity, attribute, lang_code,
        token_id_val, age_val, gender_val, nationality_val, region_val, school_val, consent_val, interface_lang,
    ):
        with open("logs/skips.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "identity": identity,
                "attribute": attribute,
                "data_point_language": lang_code,
                "token_id": token_id_val,
                "age": age_val,
                "gender": gender_val,
                "nationality": nationality_val,
                "region": region_val,
                "school": school_val,
                "consent_checkbox": consent_val,
                "interface_lang": interface_lang,
            }, ensure_ascii=False) + "\n")
        pd.DataFrame([{
            "identity": identity, "attribute": attribute, "annotator_id": token_id_val,
        }]).to_csv(f"logs/skips_{lang_code}.csv", mode="a", header=False, index=False)

    def _log_result(
        token_id_val, age_val, gender_val, nationality_val, region_val, school_val, consent_val,
        identity, attribute, lang_code, stereotype,
        assoc_nats, assoc_regions, assoc_attrs, interface_lang,
    ):
        pd.DataFrame([{
            "identity": identity, "attribute": attribute, "annotator_id": token_id_val,
        }]).to_csv(f"logs/ws_validations_{lang_code}.csv", mode="a", header=False, index=False)

        if assoc_attrs and assoc_attrs.strip():
            pd.DataFrame([{
                "identity": identity,
                "attribute": assoc_attrs.strip(),
                "annotator_id": token_id_val,
                "annotator_nationalities": nationality_val,
                "source_language": lang_code,
            }]).to_csv(f"logs/ws_stereotypes_{lang_code}.csv", mode="a", header=False, index=False)

        if assoc_nats:
            pd.DataFrame([{
                "identity": nat,
                "attribute": attribute,
                "annotator_id": token_id_val,
                "annotator_nationalities": nationality_val,
                "source_language": lang_code,
            } for nat in assoc_nats]).to_csv(
                f"logs/ws_stereotypes_{lang_code}.csv", mode="a", header=False, index=False
            )

        with open("logs/logs_validator.jsonl", "a+", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "token_id": token_id_val,
                "age": age_val,
                "gender": gender_val,
                "nationality": nationality_val,
                "region": region_val,
                "school": school_val,
                "consent_checkbox": consent_val,
                "interface_lang": interface_lang,
                "data_point": {"identity": identity, "attribute": attribute},
                "data_point_language": lang_code,
                "stereotype": stereotype,
                "associated_nationality_list": assoc_nats,
                "associated_regions_list": assoc_regions,
                "associated_attributes": assoc_attrs,
            }, ensure_ascii=False) + "\n")

    # --- Initial state ---
    initial_lang = lang.value if hasattr(lang, "value") else "es"
    initial_identity, initial_attribute, initial_dp_lang = _get_data_point(initial_lang)
    initial_choices = _get_nationality_choices(initial_lang)
    initial_identity_display = translated_nationalities[initial_lang].get(initial_identity, initial_identity)

    # --- UI ---
    with gr.Blocks() as iface:
        dp_state = gr.State([initial_identity, initial_attribute])
        dp_lang_state = gr.State(initial_dp_lang)

        gr.Markdown("# " + i18n("ValidatorTitle") + "\n\n### " + i18n("ValidatorDescription"))

        with gr.Row():
            with gr.Column(scale=1):
                data_point_box = gr.HighlightedText(
                    label=i18n("ValidatorDataPointLabel"),
                    value=[(initial_identity_display, "nationality"), (initial_attribute, "attribute")],
                    combine_adjacent=True,
                    show_legend=True,
                    interactive=False,
                    color_map=_COLOR_MAP,
                )
            with gr.Column(scale=1):
                stereotype_likert = gr.Radio(
                    [1, 2, 3, 4, 5],
                    label=i18n("ValidatorLikertLabel"),
                    info=i18n("ValidatorLikertInfo"),
                    interactive=True,
                )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                assoc_nats_dropdown = gr.Dropdown(
                    label=i18n("ValidatorAssocNationalitiesPrompt").format(attribute=initial_attribute),
                    choices=initial_choices,
                    multiselect=True,
                )
            assoc_regions_dropdown = gr.Dropdown(
                label=i18n("ValidatorAssocRegionLabel"),
                choices=[],
                allow_custom_value=True,
                multiselect=True,
                interactive=False,
                visible=False,
                scale=1,
            )

        assoc_attrs_input = gr.Textbox(
            label=i18n("ValidatorAssocAttributesPrompt").format(identity=initial_identity_display),
            placeholder=i18n("ValidatorAssocAttributesPlaceholder"),
        )

        with gr.Row():
            skip_button = gr.Button(i18n("ValidatorSkipButton"), variant="primary", scale=25)
            submit_button = gr.Button(i18n("ValidatorSubmitButton"), variant="secondary", scale=75, interactive=False)

        # --- Event handlers ---

        def _next_dp(lang_code, token_id_val, nationality_val):
            new_identity, new_attribute, new_dp_lang = _get_data_point(lang_code, token_id_val, nationality_val)
            identity_display = translated_nationalities[lang_code].get(new_identity, new_identity)
            choices = _get_nationality_choices(lang_code)
            return (
                gr.update(value=[(identity_display, "nationality"), (new_attribute, "attribute")]),
                [new_identity, new_attribute],
                new_dp_lang,
                None,
                gr.update(choices=choices, value=[], label=i18n("ValidatorAssocNationalitiesPrompt").format(attribute=new_attribute)),
                gr.update(value=[], visible=False),
                gr.update(value="", label=i18n("ValidatorAssocAttributesPrompt").format(identity=identity_display)),
                gr.update(interactive=False),
            )

        _next_outputs = [
            data_point_box,
            dp_state,
            dp_lang_state,
            stereotype_likert,
            assoc_nats_dropdown,
            assoc_regions_dropdown,
            assoc_attrs_input,
            submit_button,
        ]

        def on_submit(
            token_id_val, age_val, gender_val, nationality_val, region_val, school_val, consent_val,
            stereotype, assoc_nats, assoc_regions, assoc_attrs,
            dp, dp_lang, lang_code,
        ):
            _log_result(
                token_id_val, age_val, gender_val, nationality_val, region_val, school_val, consent_val,
                dp[0], dp[1], dp_lang, stereotype, assoc_nats, assoc_regions, assoc_attrs, lang_code,
            )
            return _next_dp(lang_code, token_id_val, nationality_val)

        def on_skip(
            token_id_val, age_val, gender_val, nationality_val, region_val, school_val, consent_val,
            dp, dp_lang, lang_code,
        ):
            if dp and len(dp) >= 2:
                _log_skip(
                    dp[0], dp[1], dp_lang,
                    token_id_val, age_val, gender_val, nationality_val, region_val, school_val, consent_val, lang_code,
                )
            return _next_dp(lang_code, token_id_val, nationality_val)

        submit_button.click(
            on_submit,
            inputs=[
                token_id, age, gender, nationality, region, school, consent_checkbox,
                stereotype_likert, assoc_nats_dropdown, assoc_regions_dropdown, assoc_attrs_input,
                dp_state, dp_lang_state, lang,
            ],
            outputs=_next_outputs,
        )

        skip_button.click(
            on_skip,
            inputs=[token_id, age, gender, nationality, region, school, consent_checkbox, dp_state, dp_lang_state, lang],
            outputs=_next_outputs,
        )

        stereotype_likert.change(
            fn=lambda val: gr.update(interactive=val is not None),
            inputs=[stereotype_likert],
            outputs=[submit_button],
        )

        assoc_nats_dropdown.change(
            fn=lambda nats: gr.update(
                choices=_get_divisions(nats),
                value=[],
                interactive=bool(nats),
                visible=bool(nats),
            ),
            inputs=[assoc_nats_dropdown],
            outputs=[assoc_regions_dropdown],
        )

    return iface
