import json
import os
from datetime import datetime

import country_converter as coco
import gradio as gr
from gradio_i18n import Translate, gettext as i18n
import pandas as pd

from interfaces.data_selection import select_data_point
from interfaces.nationalities import (
    nationalities,
    nationalities_es,
    nationalities_pt,
    translated_nationalities
)

# --- Language Handling ---
AVAILABLE_LANGUAGES = {"English": "en", "Español": "es", "Português": "pt"}
DEFAULT_LANG = "es"  # Default starting language

# --- Interface ---
def interface() -> gr.Blocks:
    # Set up country converter
    # coco.logging.getLogger().setLevel(coco.logging.CRITICAL)
    cc = coco.CountryConverter(only_UNmember=True)

    # Check for required seed data files (HESEIA per language)
    required_seed_files = {
        f"data/heseia_{lang_code}.csv": f"HESEIA dataset ({lang_code})"
        for lang_code in AVAILABLE_LANGUAGES.values()
    }
    # Check for other required general data files
    required_general_files = {
        "data/global_administrative_division.json": "Administrative divisions",
        "data/country_borders.csv": "Country borders dataset",
    }
    # Combine all required files
    required_files = {**required_seed_files, **required_general_files}

    for file_path, description in required_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Required {description} file not found: {file_path}"
            )

    # Make sure nationalities are the same length
    if len(nationalities) != len(nationalities_es) or len(nationalities) != len(nationalities_pt):
        raise ValueError(f"Error in nationalities: Length mismatch between English and other languages.")

    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Define path for JSONL skip log (remains singular)
    skip_log_path = "logs/skips.jsonl"

    # Create language-specific empty CSV dataframes if they don't exist
    for lang_code in AVAILABLE_LANGUAGES.values():
        ws_stereotypes_path_lang = f"logs/ws_stereotypes_{lang_code}.csv"
        ws_validations_path_lang = f"logs/ws_validations_{lang_code}.csv"
        skip_csv_path_lang = f"logs/skips_{lang_code}.csv"

        if not os.path.exists(ws_stereotypes_path_lang):
            pd.DataFrame(
                columns=["identity", "attribute",
                        "annotator_id", "annotator_nationalities", "source_language"]
            ).to_csv(ws_stereotypes_path_lang, index=False)

        if not os.path.exists(ws_validations_path_lang):
            pd.DataFrame(columns=["identity", "attribute", "annotator_id"]).to_csv(
                ws_validations_path_lang, index=False
            )

        if not os.path.exists(skip_csv_path_lang):
            pd.DataFrame(columns=["identity", "attribute", "annotator_id"]).to_csv(
                skip_csv_path_lang, index=False
            )

    # Load required border dataset
    df_borders = pd.read_csv("data/country_borders.csv")

    # Helper function to get administrative divisions
    def get_administrative_divisions(selected_countries):
        """Fetches administrative divisions for selected countries."""
        try:
            df = pd.read_json("data/global_administrative_division.json")
            filtered_df = df[df["name"].isin(selected_countries)]
            # Format as "Division Name (Country Name)"
            divisions = [
                f"{division['name']} ({row['name']})"
                for _, row in filtered_df.iterrows()
                for division in row["AD"]
            ]
            return sorted(list(set(divisions))) # Sort and remove duplicates
        except FileNotFoundError:
            print("Error: data/global_administrative_division.json not found.")
            return []
        except Exception as e:
            print(f"Error reading or processing administrative divisions: {e}")
            return []

    def log_skip(identity, attribute, annotator_id, data_point_language=None):
        """
        Log a skipped data point to the JSONL file and update skip counts in the language-specific CSV.

        Args:
            identity: The identity part of the skipped data point.
            attribute: The attribute part of the skipped data point.
            annotator_id: ID of the annotator who skipped the data point.
        """
        # Create skip data record
        skip_data = {
            "identity": identity,
            "attribute": attribute,
            "annotator_id": annotator_id,
            "timestamp": datetime.now().isoformat(),
        }

        # Append to JSONL file
        with open(skip_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(skip_data, ensure_ascii=False) + "\n")

        # Determine the target language code based on the data point's language
        lang_code = data_point_language if data_point_language and data_point_language in AVAILABLE_LANGUAGES.values() else 'en'
        skip_csv_path = f"logs/skips_{lang_code}.csv"
        print(f"[log_skip] Logging skip to: {skip_csv_path}") # DEBUG PRINT

        # Append to the language-specific CSV file
        skip_entry = pd.DataFrame(
            [
                {
                    "identity": identity,
                    "attribute": attribute,
                    "annotator_id": annotator_id,
                }
            ]
        )
        skip_entry.to_csv(skip_csv_path, mode="a", header=False, index=False) # Use dynamic path

    def get_random_data_point(token_id=None, nationality_personal_info=None, understood_language_names=None):
        # Determine which HESEIA datasets to load based on understood languages
        if not understood_language_names:
            # Default to English if no languages are selected
            lang_codes_to_load = ['en']
            print("[get_random_data_point] No understood languages selected, defaulting to English.")
        else:
            # Map language names (e.g., "English") to codes (e.g., "en")
            lang_codes_to_load = [AVAILABLE_LANGUAGES.get(name) for name in understood_language_names if name in AVAILABLE_LANGUAGES]
            if not lang_codes_to_load: # Handle case where selection might be invalid somehow
                lang_codes_to_load = ['en']
                print("[get_random_data_point] Invalid language names selected, defaulting to English.")
            else:
                print(f"[get_random_data_point] Loading HESEIA for languages: {lang_codes_to_load}")

        # Load and concatenate the selected HESEIA datasets
        heseia_dfs = []
        for lang_code in lang_codes_to_load:
            heseia_path = f"data/heseia_{lang_code}.csv"
            try:
                df_lang = pd.read_csv(heseia_path)
                df_lang['source_language'] = lang_code # Add source language column
                heseia_dfs.append(df_lang)
                print(f"[get_random_data_point] Loaded {heseia_path} (shape: {df_lang.shape})")
            except FileNotFoundError:
                print(f"Warning: HESEIA file not found: {heseia_path}. Skipping.")
            except Exception as e:
                print(f"Warning: Error loading {heseia_path}: {e}. Skipping.")

        if not heseia_dfs:
            # Critical fallback: if no HESEIA files could be loaded at all, load English or raise error
            print("Critical Warning: No HESEIA data could be loaded. Attempting to load English as fallback.")
            try:
                df_heseia = pd.read_csv("data/heseia_en.csv")
                df_heseia['source_language'] = 'en' # Add source language for fallback
            except Exception as e:
                raise RuntimeError(f"CRITICAL ERROR: Could not load any HESEIA data, including fallback English: {e}")
        else:
            df_heseia = pd.concat(heseia_dfs, ignore_index=True)
            print(f"[get_random_data_point] Concatenated HESEIA data shape: {df_heseia.shape}")


        # Load stereotype logs based on the languages the user understands (same as HESEIA)
        stereotype_dfs = []
        print(f"[get_random_data_point] Loading stereotypes for languages: {lang_codes_to_load}")
        for lang_code in lang_codes_to_load:
            stereotype_path = f"logs/ws_stereotypes_{lang_code}.csv"
            try:
                df_stereotype_lang = pd.read_csv(stereotype_path)
                stereotype_dfs.append(df_stereotype_lang)
                print(f"[get_random_data_point] Loaded {stereotype_path} (shape: {df_stereotype_lang.shape})")
            except FileNotFoundError:
                print(f"Info: Stereotype file not found: {stereotype_path}. Skipping.")
            except Exception as e:
                print(f"Warning: Error loading {stereotype_path}: {e}. Skipping.")

        if stereotype_dfs:
            df_ws_stereotypes = pd.concat(stereotype_dfs, ignore_index=True)
            print(f"[get_random_data_point] Concatenated stereotype data shape: {df_ws_stereotypes.shape}")
        else:
            print("Warning: No stereotype logs could be loaded. Using empty DataFrame.")
            # Ensure the empty DataFrame includes the source_language column
            df_ws_stereotypes = pd.DataFrame(columns=["identity", "attribute", "annotator_id", "annotator_nationalities", "source_language"])


        # Load validation logs based on the languages the user understands
        validation_dfs = []
        print(f"[get_random_data_point] Loading validations for languages: {lang_codes_to_load}")
        for lang_code in lang_codes_to_load:
            validation_path = f"logs/ws_validations_{lang_code}.csv"
            try:
                df_validation_lang = pd.read_csv(validation_path)
                validation_dfs.append(df_validation_lang)
                print(f"[get_random_data_point] Loaded {validation_path} (shape: {df_validation_lang.shape})")
            except FileNotFoundError:
                print(f"Info: Validation file not found: {validation_path}. Skipping.")
            except Exception as e:
                print(f"Warning: Error loading {validation_path}: {e}. Skipping.")

        if validation_dfs:
            df_ws_validations = pd.concat(validation_dfs, ignore_index=True)
            print(f"[get_random_data_point] Concatenated validation data shape: {df_ws_validations.shape}")
        else:
            print("Warning: No validation logs could be loaded. Using empty DataFrame.")
            df_ws_validations = pd.DataFrame(columns=["identity", "attribute", "annotator_id"])


        # Load skip logs based on the languages the user understands
        skip_dfs = []
        print(f"[get_random_data_point] Loading skips for languages: {lang_codes_to_load}")
        for lang_code in lang_codes_to_load:
            skip_path = f"logs/skips_{lang_code}.csv"
            try:
                df_skip_lang = pd.read_csv(skip_path)
                skip_dfs.append(df_skip_lang)
                print(f"[get_random_data_point] Loaded {skip_path} (shape: {df_skip_lang.shape})")
            except FileNotFoundError:
                print(f"Info: Skip file not found: {skip_path}. Skipping.")
            except Exception as e:
                print(f"Warning: Error loading {skip_path}: {e}. Skipping.")

        if skip_dfs:
            df_skips = pd.concat(skip_dfs, ignore_index=True)
            print(f"[get_random_data_point] Concatenated skip data shape: {df_skips.shape}")
        else:
            print("Info: No skip logs could be loaded. Proceeding without skip data.")
            df_skips = None # select_data_point handles None skips


        # Call the function from data_selection.py with the potentially combined HESEIA data
        # select_data_point now returns (identity, attribute, language_code)
        return select_data_point(
            df_ws_stereotypes=df_ws_stereotypes,
            df_ws_validations=df_ws_validations,
            df_borders=df_borders,
            df_heseia=df_heseia, # Pass the combined dataframe
            df_seegull=None, # SeeGULL not used currently
            df_skips=df_skips,
            annotator_id=token_id,
            annotator_nationalities=nationality_personal_info,
        )

    def log_result(
        token_id,
        age,
        gender,
        nationality_personal_info,
        consent_checkbox,
        data_point, # This is the English [identity, attribute] pair from state
        stereotype,
        associated_nationality_list,
        associated_regions_list,
        associated_attributes,
        understood_languages,
        associated_attribute_language=None,
        data_point_language=None, # Add data_point_language parameter
        personal_regions_list=None, # Add new parameter for personal regions
        social_groups_input=None # Add new parameter for social groups  
    ):
        # Extract the identity and attribute from data_point correctly
        # data_point here is the English version stored in current_data_point_state
        identity, attribute = data_point[0]["token"], data_point[1]["token"]

        # Determine the language code for validation logging based on the data point's language
        validation_lang_code = data_point_language if data_point_language and data_point_language in AVAILABLE_LANGUAGES.values() else 'en'
        validation_csv_path = f"logs/ws_validations_{validation_lang_code}.csv"
        print(f"[log_result] Logging validation to: {validation_csv_path}") # DEBUG PRINT

        # Log the validation in the language-specific validation file
        validation_entry = pd.DataFrame(
            [{"identity": identity, "attribute": attribute, "annotator_id": token_id}]
        )

        # Append to the language-specific validations file
        validation_entry.to_csv(
            validation_csv_path, mode="a", header=False, index=False
        )

        # Process and save associated attributes as new stereotypes
        new_stereotypes = []
        target_lang_code = None

        # Determine the target language code using case-insensitive matching
        if associated_attribute_language:
            # Normalize input by removing whitespace and converting to lowercase
            normalized_input = associated_attribute_language.strip().lower()
            # Try to find a case-insensitive match with known languages
            for key in AVAILABLE_LANGUAGES:
                if key.lower() == normalized_input:
                    target_lang_code = AVAILABLE_LANGUAGES[key]
                    target_ws_stereotypes_path = f"logs/ws_stereotypes_{target_lang_code}.csv"
                    print(f"[log_result] Logging new attribute stereotype to: {target_ws_stereotypes_path}") # DEBUG PRINT
                    break

            # If no match was found
            if target_lang_code is None:
                print(f"[log_result] Custom or missing language '{associated_attribute_language}'. Will not log attribute stereotype to language-specific CSV.")
        else:
            # If language is missing, we won't log to a specific CSV
            print(f"[log_result] No language specified. Will not log attribute stereotype to language-specific CSV.")

        # Process associated attributes for the given nationality
        if associated_attributes and isinstance(associated_attributes, str):
            single_attribute = associated_attributes.strip()
            if single_attribute:
                # Determine the source language code to store in the stereotype entry
                source_lang_for_entry = target_lang_code if target_lang_code else associated_attribute_language

                new_stereotypes.append(
                    {
                        "identity": identity,
                        "attribute": single_attribute,
                        "annotator_id": token_id,
                        "annotator_nationalities": nationality_personal_info,
                        "source_language": source_lang_for_entry,
                    }
                )

        # Process associated nationalities for the given attribute
        new_nationality_stereotypes = [] # Separate list for nationality stereotypes
        if associated_nationality_list and len(associated_nationality_list) > 0:
            # Determine the target language code based on the data point's language
            dp_lang_code = data_point_language if data_point_language and data_point_language in AVAILABLE_LANGUAGES.values() else 'en'
            dp_ws_stereotypes_path = f"logs/ws_stereotypes_{dp_lang_code}.csv"
            print(f"[log_result] Logging new nationality stereotype to: {dp_ws_stereotypes_path}") # DEBUG PRINT

            for nat in associated_nationality_list:
                new_nationality_stereotypes.append(
                    {
                        "identity": nat, # New identity (nationality)
                        "attribute": attribute, # Shown attribute
                        "annotator_id": token_id,
                        "annotator_nationalities": nationality_personal_info,
                        "source_language": dp_lang_code,
                    }
                )
            # Save these nationality stereotypes to the data point's language file
            if new_nationality_stereotypes:
                pd.DataFrame(new_nationality_stereotypes).to_csv(
                    dp_ws_stereotypes_path, mode="a", header=False, index=False
                )


        # Save new attribute stereotypes (from the first part) ONLY if the language is predefined
        if new_stereotypes: # This list now only contains (shown_identity, new_attribute) pairs
            # Check if the selected language is one of the predefined ones before saving to CSV
            if target_lang_code: # Only proceed if target_lang_code was set (i.e., language is known)
                pd.DataFrame(new_stereotypes).to_csv(
                    target_ws_stereotypes_path, mode="a", header=False, index=False # Use dynamic path from first part
                )
            # No else needed here, the print statement moved up

        # Log everything to the main JSONL file (remains singular)
        result = {
            "timestamp": datetime.now().isoformat(),
            "token_id": token_id,
            "age": age,
            "gender": gender,
            "nationality_personal_info": nationality_personal_info,
            "personal_regions_list": personal_regions_list, # Log the new field
            "social_groups_input": social_groups_input, # Log the new field
            "consent_checkbox": consent_checkbox,
            "data_point": data_point, # Log the English version
            "data_point_language": data_point_language, # Log the original language
            "stereotype": stereotype,
            "associated_nationality_list": associated_nationality_list,
            "associated_regions_list": associated_regions_list,
            "associated_attributes": associated_attributes,
            "understood_languages": understood_languages,
            "associated_attribute_language": associated_attribute_language, # Log the new field
        }
        with open("logs/logs_validator.jsonl", "a+", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Get initial data point using default language (English)
    initial_identity, initial_attribute, initial_language = get_random_data_point(understood_language_names=["English"])

    # Helper function to update input labels based on current data point
    def update_input_labels(identity, attribute):
        return (
            i18n("associated_attributes_prompt").format(identity=identity),
            i18n("associated_nationalities_prompt").format(
                attribute=attribute),
        )

    # Get initial labels (will be set later in the Gradio Blocks definition)
    initial_attr_label, initial_nat_label = update_input_labels(initial_identity, initial_attribute)

    # Gradio interface
    # Get initial labels for dynamic fields before building the UI
    # The variables initial_attr_label and initial_nat_label are already set correctly above.
   
    with gr.Blocks() as interface:
        lang = gr.Radio(
            choices=[
                (i18n("English"), "en"),
                (i18n("Spanish"), "es"),
                (i18n("Portuguese"), "pt"),
            ],
            label=i18n("LanguageLabel"),
        )
        with Translate(
            "language/i18n_validator.json",
            lang,
            placeholder_langs=["en", "pt", "es"],
        ):
            # Language selectors at the top of the interface
            with gr.Row():
                token_id = gr.Textbox(
                    label=i18n("identifier_label"),
                    lines=1,
                    interactive=True,
                )
                understood_languages = gr.CheckboxGroup(
                    label=i18n("understood_languages_label"),
                    choices=list(AVAILABLE_LANGUAGES.keys()),
                    interactive=True,
                    # elem_id="understood_languages_checkbox",
                    scale=1
                )
                with gr.Column():
                    consent_checkbox = gr.Checkbox(
                        label=i18n("consent_label"),
                        value=False,
                        interactive=True,
                    )
                    consent_link_html = gr.HTML(
                        value=f"<a href='https://docs.google.com/document/d/18OULBvUTrF9ka_XfARHCT-xath-QCmmB2DkK3zgQUJ8/' style='color:gray'>Link 🔗</a>",
                        elem_id="consent_link_html",
                    )
            # Personal information row
            with gr.Row():
                age = gr.Number(
                    value=0,
                    label=i18n("age_label"),
                    visible=True,
                    interactive=True,
                )
                gender = gr.Radio(
                    # Assuming M/F/X are universal codes, otherwise these need translation too
                    ["M", "F", "X"],
                    label=i18n("gender_label"),
                    value=None,
                    visible=True,
                    interactive=True,
                )
                nationality_personal_info = gr.Dropdown(
                    label=i18n("nationality_label"),
                    info=i18n("nationality_info"),
                    # Use the helper function to generate choices with (label, value) pairs
                    choices=nationalities,
                    multiselect=True,
                    allow_custom_value=False,
                    interactive=True,
                )
                personal_region_dropdown = gr.Dropdown(
                    label=i18n("personal_region_label"), # Use new label
                    choices=[], # Initially empty
                    allow_custom_value=True,
                    multiselect=True,
                    interactive=False, # Initially disabled
                    scale=1, # Adjust scale as needed, matching nationality dropdown perhaps
                    visible=True
                )
                social_groups_input = gr.Textbox(
                    label=i18n("social_groups_label"),
                    info=i18n("social_groups_info"),
                    lines=1,
                    scale=1,
                    interactive=True,
                )

            gr.HTML("<hr>")

            # Create Markdown components within their proper context
            with gr.Column(visible=True) as personal_data_missing:
                personal_data_missing_md = gr.Markdown(
                    i18n("personal_data_missing_md"))
            with gr.Column(visible=False, elem_id="col") as validator_col:
                welcome_md = gr.Markdown(i18n("welcome_md"))
                with gr.Row():
                    with gr.Column(scale=1):
                        data_point_box = gr.HighlightedText(
                            label=i18n("data_point_label"),
                            value=[
                                (initial_identity, "nationality"),
                                (initial_attribute, "attribute"),
                            ],
                            combine_adjacent=True,
                            show_legend=True,
                            interactive=False,
                            color_map={"nationality": "red", "attribute": "green"},  # Use dynamic map
                        )
                    with gr.Column(scale=1):
                        stereotype_likert = gr.Radio(
                            [1, 2, 3, 4, 5],  # Assuming numbers are universal
                            label=i18n("likert_label"),
                            info=i18n("likert_info"),
                            interactive=True,
                        )
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        associated_nationalities_dropdown = gr.Dropdown(
                            label=initial_nat_label,  # Already dynamically set
                            # Use the helper function to generate choices with (label, value) pairs
                            choices=nationalities,
                            multiselect=True,
                            interactive=True,
                        )
                    associated_region_dropdown = gr.Dropdown(
                        label=i18n("associated_region_label"),
                        choices=[],
                        allow_custom_value=True,
                        multiselect=True,
                        interactive=False,
                        scale=1,
                        visible=False, # Initially hidden
                    )
                with gr.Row(equal_height=True):
                    # Adjust scale for Textbox and add new Dropdown in the same row
                    with gr.Column(scale=3): # Make Textbox wider
                        associated_attributes_input = gr.Textbox(
                            label=initial_attr_label,  # Already dynamically set
                            placeholder=i18n("associated_attributes_placeholder"),
                            interactive=True,
                        )
                    # with gr.Column(scale=1): # Add Dropdown column
                    #     # Define the new dropdown component
                    #     # Find the display name corresponding to the initial language code
                    #     initial_lang_name = next((key for key, val in AVAILABLE_LANGUAGES.items() if val == lang), "English")
                    #     associated_attribute_language_dropdown = gr.Dropdown(
                    #         label=i18n("associated_attribute_language_label"),
                    #         choices=list(AVAILABLE_LANGUAGES.keys()), # Use display names like "English", "Español"
                    #         value=initial_lang_name, # Set initial default value
                    #         interactive=True,
                    #         allow_custom_value=True, # Allow custom language input
                    #         visible=False,
                    #     )
                with gr.Row(equal_height=True):
                    skip_button = gr.Button(
                        i18n("skip_button_label"), variant="primary", scale=25
                    )
                    submit_button = gr.Button(
                        i18n("submit_button_label"), variant="secondary", scale=75
                    )

            def on_submit(
                token_id,
                age,
                gender,
                nationality_personal_info,
                consent_checkbox,
                data_point,
                stereotype,
                associated_nationality_list,
                associated_regions_list, # This is for the *associated* regions, not personal
                associated_attributes,
                understood_languages,
                personal_regions_list, # Add the new personal region dropdown value
                social_groups_input,
                lang
            ):
                data_point_for_log = [{"token": data_point[0]['token']}, {"token": data_point[1]['token']}]

                # Log using the English identity/attribute from state
                # TODO: Pass the actual language of the logged data point later
                log_result(
                    token_id,
                    age,
                    gender,
                    nationality_personal_info,
                    consent_checkbox,
                    data_point_for_log, # Log with English identity/attribute
                    stereotype,
                    associated_nationality_list,
                    associated_regions_list,
                    associated_attributes,
                    understood_languages,
                    data_point_language=lang, # Pass language from state
                    personal_regions_list=personal_regions_list, # Pass the new personal regions
                    social_groups_input=social_groups_input # Pass the social groups input
                )
                # Get new data point based on currently understood languages
                new_identity, new_attribute, new_language = get_random_data_point(
                    token_id=token_id,
                    nationality_personal_info=nationality_personal_info,
                    understood_language_names=understood_languages # Pass selected languages
                )

                # Update the input labels with new data point values (using translated identity for prompt format)
                new_attr_label, new_nat_label = update_input_labels(
                    new_identity,
                    new_attribute,
                )

                return (
                    # Update displayed value with translated identity and dynamic keys
                    gr.update(
                        value=[
                            (new_identity, "nationality"),
                            (new_attribute, "attribute"),
                        ],
                    ),
                    None,  # Clear likert
                    [],  # Clear nationalities dropdown
                    [],  # Clear regions dropdown
                    "",  # Clear attributes input
                    gr.update(label=new_attr_label), # Update attribute label
                    gr.update(label=new_nat_label), # Update nationality label
                )

            submit_button.click(
                on_submit,
                inputs=[
                    token_id,
                    age,
                    gender,
                    nationality_personal_info,
                    consent_checkbox,
                    data_point_box,
                    stereotype_likert,
                    associated_nationalities_dropdown,
                    associated_region_dropdown,
                    associated_attributes_input,
                    understood_languages,
                    personal_region_dropdown, # Add personal region dropdown to inputs
                    social_groups_input, # Add social groups input to inputs
                    lang,
                ],
                outputs=[
                    data_point_box,
                    stereotype_likert,
                    associated_nationalities_dropdown,
                    associated_region_dropdown, # Added missing output
                    associated_attributes_input, # Added missing output
                    # DO NOT clear personal_region_dropdown here
                    associated_attributes_input, # Label update target
                    associated_nationalities_dropdown, # Label update target
                ],
            )

            def on_skip(
                token_id,
                data_point,
                nationality_personal_info,
                understood_languages,
                lang,
            ):
                log_skip(
                    data_point[0]['token'],
                    data_point[1]['token'],
                    token_id,
                    data_point_language=lang)

                # Get new data point, taking skip counts into consideration and using selected languages
                new_identity, new_attribute, new_language = get_random_data_point(
                    token_id=token_id,
                    nationality_personal_info=nationality_personal_info,
                    understood_language_names=understood_languages # Pass selected languages
                )

                # Update the input labels with new data point values (using translated identity for prompt format)
                new_attr_label, new_nat_label = update_input_labels(
                    new_identity,
                    new_attribute
                )

                return (
                    # Update displayed value with translated identity and dynamic keys
                    gr.update(
                        value=[
                            (new_identity, "nationality"),
                            (new_attribute, "attribute")
                        ],
                    ),
                    None,  # Clear likert
                    [],  # Clear nationalities dropdown
                    "", # Clear attributes input
                    gr.update(label=new_attr_label), # Update attribute label
                    gr.update(label=new_nat_label), # Update nationality label
                )

            skip_button.click(
                on_skip,
                inputs=[
                    token_id,
                    data_point_box,
                    nationality_personal_info,
                    understood_languages,
                    lang,
                ],
                outputs=[
                    data_point_box,
                    stereotype_likert,
                    associated_nationalities_dropdown,
                    associated_attributes_input, # Added missing output
                    associated_attributes_input, # Label update target
                    associated_nationalities_dropdown, # Label update target
                ],
            )

            def toggle_annotation(
                token_id,
                age,
                gender,
                nationality_personal_info,
                consent_checkbox,
                understood_languages,
            ):
                is_valid = not (
                    token_id is None
                    or age is None
                    or gender is None
                    or nationality_personal_info is None
                    or consent_checkbox is None
                    or age < 0
                    or age > 100
                    or len(nationality_personal_info) == 0
                    or len(token_id) == 0
                    or not consent_checkbox
                    or understood_languages is None # Check if None
                    or len(understood_languages) == 0 # Check if empty list
                )

                if is_valid:
                    # Get a personalized data point using the user's information and selected languages
                    new_identity, new_attribute, new_language = get_random_data_point(
                        token_id=token_id,
                        nationality_personal_info=nationality_personal_info,
                        understood_language_names=understood_languages # Pass selected languages
                    )

                    # Update the input labels with new data point values (using translated identity)
                    new_attr_label, new_nat_label = update_input_labels(
                        new_identity,
                        new_attribute
                    )

                    # Return updated UI state and the new data point
                    return (
                        gr.Column(visible=True),
                        gr.Column(visible=False),
                        gr.update(
                            value=[
                                (new_identity, "nationality"),
                                (new_attribute, "attribute"),
                            ],
                        ),
                        gr.update(label=new_attr_label),
                        gr.update(label=new_nat_label),
                    )
                else:
                    # Return original UI state without changing data point, keep existing data point value
                    return (
                        gr.Column(visible=False),
                        gr.Column(visible=True),
                        gr.update(),  # Keep current data_point_box value
                        gr.update(),  # Keep current associated_attributes_input label
                        gr.update(),  # Keep current associated_nationalities_dropdown label
                    )

            toggle_annotation_inputs = [
                token_id,
                age,
                gender,
                nationality_personal_info,
                consent_checkbox,
                understood_languages,
            ]
            toggle_annotation_outputs = [
                validator_col,
                personal_data_missing,
                data_point_box,
                associated_attributes_input,
                associated_nationalities_dropdown,
            ]

            for component in toggle_annotation_inputs:
                component.change(
                    fn=toggle_annotation,
                    inputs=toggle_annotation_inputs,
                    outputs=toggle_annotation_outputs
                )

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
            nationality_personal_info.change(
                fn=update_personal_regions,
                inputs=[nationality_personal_info],
                outputs=[personal_region_dropdown]
            )

            # # Function to update the ASSOCIATED region dropdown based on selected ASSOCIATED nationalities
            # def toggle_and_update_associated_regions(associated_nationalities_list):
            #     if not associated_nationalities_list:
            #         # Disable and clear if no associated nationalities are selected
            #         return gr.update(choices=[], value=[], interactive=False)
            #     else:
            #         # Get divisions using the helper function
            #         region_choices = get_administrative_divisions(associated_nationalities_list)
            #         # Enable and update choices, clear previous selection
            #         return gr.update(choices=region_choices, value=[], interactive=True)

            # # Connect the associated nationality dropdown to update the associated region dropdown
            # associated_nationalities_dropdown.change(
            #     fn=toggle_and_update_associated_regions, # Renamed function for clarity
            #     inputs=[associated_nationalities_dropdown],
            #     outputs=[associated_region_dropdown],
            # )
        
        return interface
