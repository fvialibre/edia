# --- Imports libs ---
import configparser
import json
import os
from datetime import datetime

import gradio as gr
from gradio_i18n import Translate, gettext as i18n
import pandas as pd

# --- Imports Constants ---
from auth import SCHOOL_LIST
from html_constants import FOOTER_HTML, NAVBAR_HTML, css
# from interfaces.interface_arena import interface as interface_arena
# from interfaces.interface_biasPhrase import interface as interface_biasPhrase
# from interfaces.interface_BiasWordExplorer import \
#     interface as interface_biasWordExplorer
from interfaces.interface_chatbot import interface as interface_chatbot
# from interfaces.interface_clinicalChatbot import interface as interface_clinicalChatbot
# from interfaces.interface_data import interface as interface_data
# from interfaces.interface_logsData import interface as interface_logsData
from interfaces.interface_validator import interface as interface_validator
# from interfaces.interface_cvqa import interface as interface_cvqa
from interfaces.interface_typicalPhrases import interface as interface_typicalPhrases
from interfaces.interface_ambiguousReferences import interface as interface_ambiguousReferences
# --- Imports interfaces ---
# from interfaces.interface_WordExplorer import \
#     interface as interface_wordExplorer
# --- Imports modules ---
# from modules.model_embbeding import Embedding
# from modules.module_languageModel import LanguageModel
# from modules.module_vocabulary import Vocabulary
from modules.utils import parse_cmd_line_args
from data.nationalities import nationalities


# --- Tool config ---
cmd_line_args = parse_cmd_line_args()
cfg = configparser.ConfigParser()
cfg.read('tool.cfg')

LANGUAGE                = cfg['INTERFACE']['language']
EMBEDDINGS_PATH         = cfg['WORD_EXPLORER']['embeddings_path']
NN_METHOD               = cfg['WORD_EXPLORER']['nn_method']
MAX_NEIGHBORS           = int(cfg['WORD_EXPLORER']['max_neighbors'])
CONTEXTS_DATASET        = cfg['DATA']['contexts_dataset']
VOCABULARY_SUBSET       = cfg['DATA']['vocabulary_subset']
AVAILABLE_WORDCLOUD     = cfg['DATA'].getboolean('available_wordcloud')
SPANISH_LANGUAGE_MODEL  = cfg['LMODEL']['spanish_language_model']
ENGLISH_LANGUAGE_MODEL  = cfg['LMODEL']['english_language_model']
AVAILABLE_LOGS          = cfg['LOGS'].getboolean('available_logs')
EDIA_THEME              = gr.themes.Base.from_hub('guidoivetta/edia-theme')

# Server
QUEUE_MAX_SIZE       = int(cfg['SERVER']['queue_max_size']) if cfg['SERVER']['queue_max_size'] != 'None' else None
DEFAULT_CONCURRENCY_LIMIT = int(cfg['SERVER']['default_concurrency_limit']) if cfg['SERVER']['default_concurrency_limit'] != 'None' else None
MAX_THREADS          = int(cfg['SERVER']['max_threads'])



# # --- Main App ---
with gr.Blocks(theme=EDIA_THEME, css=css, title="EDIA") as iface:
    _ = gr.HTML(NAVBAR_HTML)
    lang = gr.Radio(
        choices=[
            ("English", "en"),
            ("Español", "es"),
            ("Português", "pt"),
        ],
        value="es",
        label="Interface Language / Idioma de Interfaz / Idioma da Interface",
    )
    with Translate(
        "language/i18n.yaml",
        lang,
        placeholder_langs=["en", "pt", "es"],
    ):
        with gr.Row(equal_height=False, variant="panel"):
            with gr.Column(scale=1, min_width=200):
                with gr.Group():
                    token_id = gr.Textbox(
                        label=i18n("Identifier"),
                        lines=1,
                    )
                    age = gr.Number(
                        value=0,
                        label=i18n("AgeLabel"),
                        visible=True,
                    )
                    gender = gr.Radio(
                        ["M", "F", "X"],
                        label=i18n("GenderLabel"),
                        value=None,
                        visible=True,
                    )
            with gr.Column(scale=2):
                with gr.Group():
                    nationality = gr.Dropdown(
                        label=i18n("NationalityLabel"),
                        info=i18n("NationalityInfo"),
                        choices=nationalities,
                        multiselect=False,
                        allow_custom_value=False,
                    )
                    region = gr.Dropdown(
                        label=i18n("PersonalRegionLabel"),
                        choices=[],
                        allow_custom_value=True,
                        multiselect=False,
                        interactive=False,
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            school = gr.Number(
                                label=i18n("SchoolLabel"),
                                visible=False,
                            )
                            school_list_link = gr.HTML(
                                value="<a href='https://docs.google.com/spreadsheets/d/1SQaQqXh46_J_VrcHo3YJUfPSfKIjbKi73EEtaImzk9c/edit'>Lista de escuelas 🔗</a>",
                                visible=False,
                            )
                            school_name = gr.HTML(
                                value=i18n("SchoolNamePlaceholder"),
                                visible=False,
                            )
            with gr.Column(scale=1, min_width=200):
                consent_checkbox = gr.Checkbox(
                    label=i18n("ConsentLabel"),
                    value=False,
                )
                _ = gr.HTML(
                    value=f"<a href='https://docs.google.com/document/d/17Feum83dTqjcicgJxuWdZ3qLuL3emmVY2idGym_usLU/edit?usp=sharing'>Link 🔗</a>",
                )
        _ = gr.HTML(
            value="<hr>",
        )

        with gr.Tabs(visible=False) as annotation_tabs:
            with gr.Tab(i18n("TypicalPhrasesTab")):
                interface_typicalPhrases(
                    token_id=token_id,
                    age=age,
                    gender=gender,
                    nationality=nationality,
                    region=region,
                    school=school,
                    consent_checkbox=consent_checkbox,
                )
            with gr.Tab(i18n("ChatbotTab")):
                interface_chatbot(
                    token_id=token_id,
                    age=age,
                    gender=gender,
                    nationality=nationality,
                    region=region,
                    school=school,
                    consent_checkbox=consent_checkbox,
                )
            with gr.Tab(i18n("AmbiguousReferencesTab")):
                interface_ambiguousReferences(
                    token_id=token_id,
                    age=age,
                    gender=gender,
                    nationality=nationality,
                    region=region,
                    school=school,
                    consent_checkbox=consent_checkbox,
                )

        
        with gr.Row(visible=True) as personal_data_missing:
            gr.Markdown(
                i18n("PersonalDataMissingHeading")
            )
        _ = gr.HTML(FOOTER_HTML)

    
    def get_administrative_divisions(selected_country):
        """Fetches administrative divisions for a selected country (string)."""
        if not selected_country:
            return []

        try:
            df = pd.read_json("data/global_administrative_division.json")
            filtered_df = df[df["name"] == selected_country]

            # Format as "Division Name (Country Name)"
            divisions = [
                f"{division['name']} ({row['name']})"
                for _, row in filtered_df.iterrows()
                for division in row["AD"]
            ]
            return sorted(list(set(divisions)))  # Sort and remove duplicates
        except FileNotFoundError:
            print("Error: data/global_administrative_division.json not found.")
            return []
        except Exception as e:
            print(f"Error reading or processing administrative divisions: {e}")
            return []

    # Function to update the PERSONAL region dropdown based on selected PERSONAL nationalities
    def update_personal_regions(selected_personal_nationalities):
        if not selected_personal_nationalities:
            # Disable and clear if no nationalities are selected
            return gr.update(choices=[], value=[], interactive=False)
        else:
            # Get divisions using the helper function
            region_choices = get_administrative_divisions(selected_personal_nationalities)
            # Enable and update choices, keep existing selection if possible (Gradio handles this)
            return gr.update(choices=region_choices, interactive=True)

    # Connect the personal nationality dropdown to update the personal region dropdown
    nationality.change(
        fn=update_personal_regions,
        inputs=[nationality],
        outputs=[region]
    )

    def toggle_school(region):
        if region == "Cordoba (Argentina)":
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    # Connect the personal nationality dropdown to update the personal region dropdown
    region.change(
        fn=toggle_school,
        inputs=[region],
        outputs=[school, school_list_link, school_name]
    )

    def update_school_name(school):
        if school is None or school == 0:
            return (
                gr.HTML(
                    value=f"<p>No seleccionaste ningún colegio</p>",
                )
            )
        elif school not in SCHOOL_LIST:
            return (
                gr.HTML(
                    value=f"<p>El colegio seleccionado no existe</p>",
                )
            )
        else:
            return (
                gr.HTML(
                    value=f"<p>Seleccionaste: {SCHOOL_LIST[school]}</p>",
                )
            )
    
    school.change(
        fn=update_school_name,
        inputs=[school],
        outputs=[school_name]
    )

    def toggle_annotation(
        token_id, age, gender, nationality, region, school, consent_checkbox
    ):
        if any([
            token_id is None,
            age is None,
            gender is None,
            nationality is None or len(nationality) == 0,
            region is None or len(region) == 0,
            region == "Cordoba (Argentina)" and school not in SCHOOL_LIST,
            consent_checkbox is None,
            age < 0,
            age > 100,
            len(token_id) == 0,
            not consent_checkbox
        ]):
            return (
                gr.Tabs(visible=False),
                gr.Row(visible=True)
            )
        else:
            return (
                gr.Tabs(visible=True),
                gr.Row(visible=False)
            )

    toggle_annotation_inputs = [
        token_id,
        age,
        gender,
        nationality,
        region,
        school,
        consent_checkbox
    ]
    toggle_annotation_outputs = [
        annotation_tabs,
        personal_data_missing
    ]

    for component in toggle_annotation_inputs:
        component.change(
            fn=toggle_annotation,
            inputs=toggle_annotation_inputs,
            outputs=toggle_annotation_outputs
        )

iface.queue(
    max_size=QUEUE_MAX_SIZE,
    default_concurrency_limit=DEFAULT_CONCURRENCY_LIMIT
)

iface.launch(
    server_port=cmd_line_args['port'],
    max_threads = MAX_THREADS
)
