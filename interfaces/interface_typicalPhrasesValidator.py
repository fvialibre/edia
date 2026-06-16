import html as _html
import json
import os
import random
from datetime import datetime

import gradio as gr
from gradio_i18n import gettext as i18n


# --- Interface ---

def interface(
    token_id,
    age,
    gender,
    nationality,
    region,
    school,
    consent_checkbox,
) -> gr.Blocks:

    LOG_FILE = "logs/df_ft_11-06-2026-vf.jsonl"
    VALIDATOR_LOG_FILE = "logs/logs_typicalPhrasesValidator.jsonl"

    def _phrase_html(text):
        s = _html.escape(text)
        return (f'<p style="font-size:1.3rem;font-weight:800;color:#3730a3;'
                f'background:#eef2ff;padding:14px 18px;border-radius:8px;margin:0">{s}</p>')

    def _definition_html(text):
        s = _html.escape(text)
        return (f'<p style="font-size:1.3rem;color:#166534;'
                f'background:#f0fdf4;padding:12px 16px;border-radius:8px;margin:0">{s}</p>')

    def _sentence_html(text):
        s = _html.escape(text)
        return (f'<p style="font-size:1.3rem;font-style:italic;color:#92400e;'
                f'background:#fffbeb;padding:12px 16px;border-radius:8px;margin:0">{s}</p>')

    # ── Data helpers ─────────────────────────────────────────────────────────

    def _sample_datapoint(current_token_id):
        """Return a random entry from LOG_FILE whose token_id differs from current_token_id."""
        try:
            entries = []
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (
                        entry.get("token_id") != current_token_id
                        and entry.get("new_phrase")
                        and entry.get("new_phrase_definition")
                        and entry.get("new_phrase_sentence_example")
                    ):
                        entries.append(entry)
            if not entries:
                return None
            return random.choice(entries)
        except FileNotFoundError:
            return None

    def _log_validation(
        token_id_val, age_val, gender_val, nationality_val,
        region_val, school_val, consent_val, sample,
        awareness, use_freq, hear_freq, toxicity,
        definition_correctness, alt_definition,
        sentence_correctness, alt_sentence,
    ):
        result = {
            "timestamp": datetime.now().isoformat(),
            "validator_token_id": token_id_val,
            "validator_age": age_val,
            "validator_gender": gender_val,
            "validator_nationality": nationality_val,
            "validator_region": region_val,
            "validator_school": school_val,
            "validator_consent": consent_val,
            "phrase_token_id": sample.get("token_id") if sample else None,
            "phrase_timestamp": sample.get("timestamp") if sample else None,
            "new_phrase": sample.get("new_phrase") if sample else None,
            "new_phrase_definition": sample.get("new_phrase_definition") if sample else None,
            "new_phrase_sentence_example": sample.get("new_phrase_sentence_example") if sample else None,
            "awareness": awareness,
            "use_frequency": use_freq if awareness in ("Yes", "DontKnow") else None,
            "hear_frequency": hear_freq if awareness in ("Yes", "DontKnow") else None,
            "toxicity": toxicity,
            "definition_correctness": definition_correctness,
            "alt_definition": alt_definition.strip() if alt_definition and alt_definition.strip() else None,
            "sentence_correctness": sentence_correctness,
            "alt_sentence": alt_sentence.strip() if alt_sentence and alt_sentence.strip() else None,
            "source": sample,
        }
        with open(VALIDATOR_LOG_FILE, "a+", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def _build_outputs(new_sample):
        """Shared reset tuple used by load, skip, and next."""
        if new_sample is None:
            return (
                None,
                gr.update(visible=True),          # no_data_msg
                gr.update(visible=False),         # form_col
                gr.update(value=""),              # phrase_display
                gr.update(value=""),              # definition_display
                gr.update(value=""),              # sentence_display
                gr.update(value=None),            # awareness_radio
                gr.update(value=None),            # toxicity_radio
                gr.update(value=None),            # use_freq_radio
                gr.update(value=None),            # hear_freq_radio
                gr.update(visible=False),         # freq_col
                gr.update(visible=False),         # def_sent_col
                gr.update(value=None),            # definition_correctness
                gr.update(value=None),            # sentence_correctness
                "",                               # alt_definition
                "",                               # alt_sentence
                gr.update(interactive=False),     # next_button
            )
        return (
            new_sample,
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=_phrase_html(new_sample["new_phrase"])),
            gr.update(value=_definition_html(new_sample["new_phrase_definition"])),
            gr.update(value=_sentence_html(new_sample["new_phrase_sentence_example"])),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=None),
            gr.update(value=None),
            "",
            "",
            gr.update(interactive=False),
        )

    # ── Gradio Interface ─────────────────────────────────────────────────────

    with gr.Blocks() as interface:
        current_sample = gr.State(None)

        _ = gr.Markdown(
            "# " + i18n("TypicalPhrasesValidatorTitle") + "\n\n" +
            i18n("TypicalPhrasesValidatorDescription")
        )

        no_data_msg = gr.Markdown(
            i18n("TypicalPhrasesValidatorNoData"),
            visible=False,
        )

        with gr.Column(visible=True) as form_col:

            # ── Section 1: Phrase + awareness ────────────────────────────────
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, variant="panel"):
                    gr.Markdown("#### " + i18n("TypicalPhrasesValidatorPhraseSection"))
                    phrase_display = gr.Markdown("", sanitize_html=False)
                with gr.Column(scale=5, variant="panel"):
                    awareness_radio = gr.Radio(
                        choices=[
                            (i18n("TypicalPhrasesValidatorAwarenessNo"), "No"),
                            (i18n("TypicalPhrasesValidatorAwarenessDontKnow"), "DontKnow"),
                            (i18n("TypicalPhrasesValidatorAwarenessYes"), "Yes"),
                        ],
                        label=i18n("TypicalPhrasesValidatorAwarenessLabel"),
                        value=None,
                        interactive=True,
                    )
                    with gr.Column(visible=False, variant="panel") as freq_col:
                        toxicity_radio = gr.Radio(
                            choices=[
                                1, 2, 3, 4, 5,
                                (i18n("TypicalPhrasesValidatorToxicityDependsContext"), "depends_on_context"),
                            ],
                            label=i18n("TypicalPhrasesValidatorToxicityLabel"),
                            info=i18n("TypicalPhrasesValidatorToxicityLikertInfo"),
                            value=None,
                            interactive=True,
                        )
                        use_freq_radio = gr.Radio(
                            choices=[
                                (i18n("FreqNever"), "Never"),
                                (i18n("FreqRarely"), "Rarely"),
                                (i18n("FreqSometimes"), "Sometimes"),
                                (i18n("FreqOften"), "Often"),
                                (i18n("FreqAllTheTime"), "All the time"),
                            ],
                            label=i18n("TypicalPhrasesValidatorUseFreqLabel"),
                            value=None,
                            interactive=True,
                        )
                        hear_freq_radio = gr.Radio(
                            choices=[
                                (i18n("FreqNever"), "Never"),
                                (i18n("FreqRarely"), "Rarely"),
                                (i18n("FreqSometimes"), "Sometimes"),
                                (i18n("FreqOften"), "Often"),
                                (i18n("FreqAllTheTime"), "All the time"),
                            ],
                            label=i18n("TypicalPhrasesValidatorHearFreqLabel"),
                            value=None,
                            interactive=True,
                        )

            # ── Section 2: Definition + questions ────────────────────────────
            with gr.Column(visible=False) as def_sent_col:
                gr.Markdown("---")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3, variant="panel"):
                        gr.Markdown("#### " + i18n("TypicalPhrasesValidatorDefinitionSection"))
                        definition_display = gr.Markdown("", sanitize_html=False)
                    with gr.Column(scale=5, variant="panel"):
                        definition_correctness = gr.Radio(
                            choices=[1, 2, 3, 4, 5],
                            label=i18n("TypicalPhrasesValidatorDefinitionCorrectnessLabel"),
                            info=i18n("TypicalPhrasesValidatorLikertInfo"),
                            value=None,
                            interactive=True,
                        )
                        alt_definition = gr.Textbox(
                            label=i18n("TypicalPhrasesValidatorAltDefinitionLabel"),
                            placeholder=i18n("TypicalPhrasesValidatorAltDefinitionPlaceholder"),
                            lines=2,
                        )

                # ── Section 3: Sentence + questions ──────────────────────────
                gr.Markdown("---")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3, variant="panel"):
                        gr.Markdown("#### " + i18n("TypicalPhrasesValidatorSentenceSection"))
                        sentence_display = gr.Markdown("", sanitize_html=False)
                    with gr.Column(scale=5, variant="panel"):
                        sentence_correctness = gr.Radio(
                            choices=[1, 2, 3, 4, 5],
                            label=i18n("TypicalPhrasesValidatorSentenceCorrectnessLabel"),
                            info=i18n("TypicalPhrasesValidatorSentenceLikertInfo"),
                            value=None,
                            interactive=True,
                        )
                        alt_sentence = gr.Textbox(
                            label=i18n("TypicalPhrasesValidatorAltSentenceLabel"),
                            placeholder=i18n("TypicalPhrasesValidatorAltSentencePlaceholder"),
                            lines=2,
                        )

            # ── Buttons ───────────────────────────────────────────────────────
            with gr.Row():
                skip_button = gr.Button(
                    i18n("SkipButton"),
                    variant="secondary",
                    interactive=True,
                )
                next_button = gr.Button(
                    i18n("NextButton"),
                    interactive=False,
                    variant="primary",
                )

        # ── Shared outputs list ──────────────────────────────────────────────

        _OUTPUTS = [
            current_sample,
            no_data_msg, form_col,
            phrase_display, definition_display, sentence_display,
            awareness_radio, toxicity_radio, use_freq_radio, hear_freq_radio, freq_col,
            def_sent_col,
            definition_correctness, sentence_correctness,
            alt_definition, alt_sentence, next_button,
        ]

        # ── Load initial sample ──────────────────────────────────────────────

        def _load_sample(token_id_val):
            return _build_outputs(_sample_datapoint(token_id_val))

        interface.load(
            _load_sample,
            inputs=[token_id],
            outputs=_OUTPUTS,
        )

        # ── Awareness change → show/hide freq + toggle Next ──────────────────

        def _on_awareness_change(awareness, use_freq, hear_freq, toxicity, def_corr, sent_corr):
            show = awareness in ("Yes", "DontKnow")
            if not show:
                use_freq = None
                hear_freq = None
            ok = (
                awareness is not None
                and (awareness == "No"
                     or (show and toxicity is not None and use_freq and hear_freq
                         and def_corr is not None and sent_corr is not None))
            )
            return (
                gr.update(visible=show),       # freq_col
                gr.update(visible=show),       # def_sent_col
                gr.update(value=use_freq),     # use_freq_radio
                gr.update(value=hear_freq),    # hear_freq_radio
                gr.update(interactive=ok),     # next_button
            )

        awareness_radio.change(
            _on_awareness_change,
            inputs=[awareness_radio, use_freq_radio, hear_freq_radio,
                    toxicity_radio, definition_correctness, sentence_correctness],
            outputs=[freq_col, def_sent_col, use_freq_radio, hear_freq_radio, next_button],
        )

        # ── Other fields change → toggle Next button ─────────────────────────

        def _check_required(awareness, use_freq, hear_freq, toxicity, def_corr, sent_corr):
            ok = (
                awareness is not None
                and (awareness == "No"
                     or (awareness in ("Yes", "DontKnow") and toxicity is not None
                         and use_freq and hear_freq
                         and def_corr is not None and sent_corr is not None))
            )
            return gr.update(interactive=ok)

        for _comp in [use_freq_radio, hear_freq_radio, toxicity_radio, definition_correctness, sentence_correctness]:
            _comp.change(
                _check_required,
                inputs=[awareness_radio, use_freq_radio, hear_freq_radio,
                        toxicity_radio, definition_correctness, sentence_correctness],
                outputs=[next_button],
            )

        # ── Skip button → sample new phrase without logging ──────────────────

        def _on_skip(token_id_val):
            return _build_outputs(_sample_datapoint(token_id_val))

        skip_button.click(
            _on_skip,
            inputs=[token_id],
            outputs=_OUTPUTS,
        )

        # ── Next button → log + sample new phrase ────────────────────────────

        def _on_next(
            token_id_val, age_val, gender_val, nationality_val,
            region_val, school_val, consent_val,
            sample,
            awareness, use_freq, hear_freq, toxicity,
            def_corr, alt_def,
            sent_corr, alt_sent,
        ):
            _log_validation(
                token_id_val, age_val, gender_val, nationality_val,
                region_val, school_val, consent_val, sample,
                awareness, use_freq, hear_freq, toxicity,
                def_corr, alt_def,
                sent_corr, alt_sent,
            )
            return _build_outputs(_sample_datapoint(token_id_val))

        next_button.click(
            _on_next,
            inputs=[
                token_id, age, gender, nationality, region, school, consent_checkbox,
                current_sample,
                awareness_radio, use_freq_radio, hear_freq_radio, toxicity_radio,
                definition_correctness, alt_definition,
                sentence_correctness, alt_sentence,
            ],
            outputs=_OUTPUTS,
        )

    return interface
