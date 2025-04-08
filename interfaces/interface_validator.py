import json
import os
from datetime import datetime

import country_converter as coco
import gradio as gr
import pandas as pd

from interfaces.data_selection import select_data_point
from interfaces.nationalities import (nationalities, nationalities_es,
                                      nationalities_pt,
                                      translated_nationalities)

# --- Language Handling ---
AVAILABLE_LANGUAGES = {"English": "en", "Español": "es", "Português": "pt"}
DEFAULT_LANG = "es"  # Default starting language

# --- Nationality Data Handling ---
NATIONALITY_DATA = {
    "en": nationalities,
    "es": nationalities_es,
    "pt": nationalities_pt,
}

def get_nationality_choices(lang_code: str):
    """Generates (label, value) tuples for nationality dropdowns."""
    # Use English as fallback if lang_code is invalid or list is missing
    translated_list = NATIONALITY_DATA.get(lang_code, nationalities)
    choices = list(zip(translated_list, nationalities))
    return choices


# --- Language Handling ---
def load_language(lang: str):
    """Loads language labels for the validator interface."""
    labels_path = f"language/{lang}.json"
    fallback_path = f"language/en.json"  # English as fallback

    if not os.path.exists(labels_path):
        print(
            f"Warning: Language file {labels_path} not found. Defaulting to English.")
        labels_path = fallback_path
        lang = "en"  # Update lang if falling back

    try:
        # Using pandas consistent with interface_crowsPairs.py
        all_labels = pd.read_json(labels_path)
        # Use a key consistent with others, e.g., "validator_interface"
        labels = all_labels["validator_interface"]
        # Add the current language code to the labels dict for reference
        labels["current_lang"] = lang
        print(f"[load_language] Loading lang: {lang}") # DEBUG PRINT
        print(f"[load_language] Loaded labels: {labels}") # DEBUG PRINT - Potentially too verbose
        return labels
    except KeyError:
        # Handle missing key - maybe load English as fallback?
        print(
            f"Warning: 'validator_interface' key not found in {labels_path}. Loading English."
        )
        all_labels = pd.read_json(fallback_path)
        try:
            labels = all_labels[
                "validator_interface"
            ]  # Assuming English file has the key
            labels["current_lang"] = "en"  # Mark as fallback lang
            return labels
        except KeyError:
            raise RuntimeError(
                f"Critical: 'validator_interface' key not found in fallback English file {fallback_path}"
            )
    except Exception as e:
        # Handle other potential errors during loading
        raise RuntimeError(f"Error loading language file {labels_path}: {e}")


# --- Interface ---
def interface(lang: str = "es") -> gr.Blocks:
    # --- Initial language load ---
    labels = load_language(lang)

    # Set up country converter
    coco.logging.getLogger().setLevel(coco.logging.CRITICAL)
    cc = coco.CountryConverter(only_UNmember=True)

    # Check for required files
    required_files = {
        "data/processed_Frases_HESEIA_Anotación.csv": "HESEIA dataset",
        "data/global_administrative_division.json": "Administrative divisions",
        "data/country_borders.csv": "Country borders dataset",
    }

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

    # Create empty dataframes if they don't exist
    ws_stereotypes_path = "logs/ws_stereotypes.csv"
    ws_validations_path = "logs/ws_validations.csv"
    skip_log_path = "logs/skips.jsonl"
    skip_csv_path = "logs/skips.csv"

    if not os.path.exists(ws_stereotypes_path):
        pd.DataFrame(
            columns=["identity", "attribute",
                     "annotator_id", "annotator_nationalities"]
        ).to_csv(ws_stereotypes_path, index=False)

    if not os.path.exists(ws_validations_path):
        pd.DataFrame(columns=["identity", "attribute", "annotator_id"]).to_csv(
            ws_validations_path, index=False
        )

    # Create skips.csv if it doesn't exist
    if not os.path.exists(skip_csv_path):
        pd.DataFrame(columns=["identity", "attribute", "annotator_id"]).to_csv(
            skip_csv_path, index=False
        )

    # Load required datasets
    df_heseia = pd.read_csv("data/processed_Frases_HESEIA_Anotación.csv")
    df_borders = pd.read_csv("data/country_borders.csv")

    def log_skip(identity, attribute, annotator_id):
        """
        Log a skipped data point to the JSONL file and update skip counts in CSV.

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

        # Append to CSV file
        skip_entry = pd.DataFrame(
            [
                {
                    "identity": identity,
                    "attribute": attribute,
                    "annotator_id": annotator_id,
                }
            ]
        )
        skip_entry.to_csv(skip_csv_path, mode="a", header=False, index=False)

    def get_random_data_point(token_id=None, nationality_personal_info=None):
        # Read the most up-to-date versions of the dataframes
        df_ws_stereotypes = pd.read_csv(ws_stereotypes_path)
        df_ws_validations = pd.read_csv(ws_validations_path)

        # Load skip counts if the file exists
        df_skips = None
        if os.path.exists(skip_csv_path):
            df_skips = pd.read_csv(skip_csv_path)

        # Call the function from data_selection.py
        return select_data_point(
            df_ws_stereotypes=df_ws_stereotypes,
            df_ws_validations=df_ws_validations,
            df_borders=df_borders,
            df_heseia=df_heseia,
            df_seegull=None,
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
        data_point,
        stereotype,
        associated_nationality_list,
        associated_regions_list,
        associated_attributes,
    ):
        # Extract the identity and attribute from data_point correctly
        identity, attribute = data_point[0]["token"], data_point[1]["token"]

        # Log the validation in the validation file
        validation_entry = pd.DataFrame(
            [{"identity": identity, "attribute": attribute, "annotator_id": token_id}]
        )

        # Append to the validations file
        validation_entry.to_csv(
            ws_validations_path, mode="a", header=False, index=False
        )

        # Process and save associated attributes as new stereotypes
        new_stereotypes = []

        # Process associated attributes for the given nationality
        if associated_attributes and isinstance(associated_attributes, str):
            single_attribute = associated_attributes.strip()
            if single_attribute:
                new_stereotypes.append(
                    {
                        "identity": identity,
                        "attribute": single_attribute,
                        "annotator_id": token_id,
                        "annotator_nationalities": nationality_personal_info,
                    }
                )

        # Process associated nationalities for the given attribute
        if associated_nationality_list and len(associated_nationality_list) > 0:
            for nat in associated_nationality_list:
                new_stereotypes.append(
                    {
                        "identity": nat,
                        "attribute": attribute,
                        "annotator_id": token_id,
                        "annotator_nationalities": nationality_personal_info,
                    }
                )

        # Save new stereotypes to the workshop stereotypes file if we have any
        if new_stereotypes:
            pd.DataFrame(new_stereotypes).to_csv(
                ws_stereotypes_path, mode="a", header=False, index=False
            )

        result = {
            "timestamp": datetime.now().isoformat(),
            "token_id": token_id,
            "age": age,
            "gender": gender,
            "nationality_personal_info": nationality_personal_info,
            "consent_checkbox": consent_checkbox,
            "data_point": data_point,
            "stereotype": stereotype,
            "associated_nationality_list": associated_nationality_list,
            "associated_regions_list": associated_regions_list,
            "associated_attributes": associated_attributes,
        }
        with open("logs/logs_validator.jsonl", "a+", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    initial_identity, initial_attribute = get_random_data_point()

    # Helper function to update input labels based on current data point
    def update_input_labels(identity, attribute, labels): # Added 'labels' argument
        return (
            labels["associated_attributes_prompt"].format(identity=identity),
            labels["associated_nationalities_prompt"].format(
                attribute=attribute),
        )

    # Get initial labels (will be set later in the Gradio Blocks definition)
    initial_attr_label, initial_nat_label = update_input_labels(initial_identity, initial_attribute, labels) # Pass initial labels

    # Get translated keys for HighlightedText
    nationality_key = labels.get(
        "data_point_legend_nationality", "nationality")
    attribute_key = labels.get("data_point_legend_attribute", "attribute")
    dynamic_color_map = {nationality_key: "red", attribute_key: "green"}

    # Gradio interface
    # Get initial labels for dynamic fields before building the UI
    # The variables initial_attr_label and initial_nat_label are already set correctly above.

    # Get initial nationality choices based on the starting language
    initial_nationality_choices = get_nationality_choices(lang)

    # Translate the initial identity for display
    initial_identity_display = translated_nationalities[lang].get(initial_identity, initial_identity)

    with gr.Blocks() as interface:
        # State to hold the current language labels
        language_labels_state = gr.State(labels)
        # State to hold the current English data point [identity, attribute]
        current_data_point_state = gr.State([initial_identity, initial_attribute])

        # Language selector at the top of the interface
        with gr.Row():
            language_dropdown = gr.Dropdown(
                label="Language / Idioma / Idioma",  # Multilingual label
                choices=list(AVAILABLE_LANGUAGES.items()),
                value=lang,  # Default to server-provided language
                interactive=True,
                elem_id="language_dropdown",
            )
            gr.HTML("<div style='flex-grow: 1'></div>")  # Spacer

        # Personal information row
        with gr.Row():
            token_id = gr.Textbox(
                label=labels["identifier_label"],
                info=labels["identifier_info"],
                lines=1,
            )
            age = gr.Number(
                value=0,
                label=labels["age_label"],
                visible=False,
            )
            gender = gr.Radio(
                # Assuming M/F/X are universal codes, otherwise these need translation too
                ["M", "F", "X"],
                label=labels["gender_label"],
                value="X",
                visible=False,
            )
            nationality_personal_info = gr.Dropdown(
                label=labels["nationality_label"],
                info=labels["nationality_info"],
                # Use the helper function to generate choices with (label, value) pairs
                choices=initial_nationality_choices,
                multiselect=True,
                allow_custom_value=False,
            )
            with gr.Column():
                consent_checkbox = gr.Checkbox(
                    label=labels["consent_label"], value=False
                )
                consent_link_html = gr.HTML(
                    value=f"<a href='https://docs.google.com/document/d/1YEi0QpFYJwFBSIAjGplPc0VkOxJwnME29dWWyfp37XY/edit?usp=sharing'>{labels['consent_link_text']}</a>",
                    elem_id="consent_link_html",
                )

        gr.HTML("<hr>")

        # Create Markdown components within their proper context
        with gr.Column(visible=True) as personal_data_missing:
            personal_data_missing_md = gr.Markdown(
                labels["personal_data_missing_md"])
        with gr.Column(visible=False, elem_id="col") as validator_col:
            welcome_md = gr.Markdown(labels["welcome_md"])
            with gr.Row():
                with gr.Column(scale=1):
                    # Assuming color_map keys 'nationality' and 'attribute' are internal identifiers
                    # If the displayed legend text needs translation, we'd need more complex setup
                    # For now, let's assume the legend text comes from the tuple values directly
                    # and the label needs translation.
                    data_point_box = gr.HighlightedText(
                        label=labels["data_point_label"],
                        value=[
                            # Use dynamic key and translated initial identity
                            (initial_identity_display, nationality_key),
                            (initial_attribute, attribute_key),  # Use dynamic key
                        ],
                        combine_adjacent=True,
                        show_legend=True,
                        interactive=False,
                        color_map=dynamic_color_map,  # Use dynamic map
                    )
                with gr.Column(scale=1):
                    stereotype_likert = gr.Radio(
                        [1, 2, 3, 4, 5],  # Assuming numbers are universal
                        label=labels["likert_label"],
                        info=labels["likert_info"],
                        interactive=True,
                    )
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    associated_attributes_input = gr.Textbox(
                        label=initial_attr_label,  # Already dynamically set
                        placeholder=labels["associated_attributes_placeholder"],
                    )
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    associated_nationalities_dropdown = gr.Dropdown(
                        label=initial_nat_label,  # Already dynamically set
                        # Use the helper function to generate choices with (label, value) pairs
                        choices=initial_nationality_choices,
                        multiselect=True,
                    )
                associated_region_dropdown = gr.Dropdown(
                    label=labels["associated_region_label"],
                    choices=[],
                    multiselect=True,
                    interactive=False,
                    scale=1
                )
            with gr.Row(equal_height=True):
                skip_button = gr.Button(
                    labels["skip_button_label"], variant="primary", scale=25
                )
                submit_button = gr.Button(
                    labels["submit_button_label"], variant="secondary", scale=75
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
            associated_regions_list,
            associated_attributes,
            current_labels,
            current_data_point
        ):
            print(f"[on_submit] Received current_labels from state: {current_labels}") # DEBUG PRINT
            print(f"[on_submit] Received current_data_point state: {current_data_point}") # DEBUG PRINT

            # Extract English identity/attribute from state for logging
            identity_en = current_data_point[0] if current_data_point else None
            attribute_en = current_data_point[1] if current_data_point and len(current_data_point) >= 2 else None

            # Create a data_point structure with English tokens for logging
            data_point_for_log = [{"token": identity_en}, {"token": attribute_en}]

            # Log using the English identity/attribute from state
            log_result(
                token_id,
                age,
                gender,
                nationality_personal_info,
                consent_checkbox,
                data_point_for_log,
                stereotype,
                associated_nationality_list,
                associated_regions_list,
                associated_attributes,
            )
            new_identity, new_attribute = get_random_data_point(
                token_id=token_id, nationality_personal_info=nationality_personal_info
            )

            # Translate the new identity for display
            current_lang = current_labels.get("current_lang", "en") # Get current lang from labels state
            new_identity_display = translated_nationalities[current_lang].get(new_identity, new_identity)

            # Update the input labels with new data point values (using translated identity for prompt format)
            new_attr_label, new_nat_label = update_input_labels(
                new_identity_display, new_attribute, current_labels # Pass current_labels and translated identity
            )

            # Get current legend keys from state
            nationality_key = current_labels.get(
                "data_point_legend_nationality", "nationality"
            )
            attribute_key = current_labels.get(
                "data_point_legend_attribute", "attribute"
            )

            color_map = {nationality_key: "red", attribute_key: "green"}

            print(f"[on_submit] Using keys: Nat='{nationality_key}', Attr='{attribute_key}'") # DEBUG PRINT
            print(f"[on_submit] New English Identity: {new_identity}, Display Identity: {new_identity_display}") # DEBUG PRINT

            return (
                # Update displayed value with translated identity and dynamic keys
                gr.update(
                    value=[(new_identity_display, nationality_key),
                           (new_attribute, attribute_key)],
                    color_map=color_map,
                ),
                # Update the state holding the English data point
                [new_identity, new_attribute],
                None,  # Clear likert
                [],  # Clear nationalities dropdown
                [],
                "",
                gr.update(label=new_attr_label),
                gr.update(label=new_nat_label),
            )

        def on_skip(
            token_id, current_data_point, nationality_personal_info, current_labels # Changed data_point input to current_data_point state
        ):  # Added state input
            print(f"[on_skip] Received current_labels from state: {current_labels}") # DEBUG PRINT
            # Extract current English identity and attribute from state
            if current_data_point and len(current_data_point) >= 2:
                identity = current_data_point[0]
                attribute = current_data_point[1]

                # Log the skip if we have valid identity and attribute
                if identity and attribute:
                    log_skip(identity, attribute, token_id)

            # Get new data point, taking skip counts into consideration
            new_identity, new_attribute = get_random_data_point(
                token_id=token_id,
                nationality_personal_info=nationality_personal_info,
            )

            # Translate the new identity for display
            current_lang = current_labels.get("current_lang", "en") # Get current lang from labels state
            new_identity_display = translated_nationalities[current_lang].get(new_identity, new_identity)

            # Update the input labels with new data point values (using translated identity for prompt format)
            new_attr_label, new_nat_label = update_input_labels(
                new_identity_display, new_attribute, current_labels # Pass current_labels and translated identity
            )

            # Get current legend keys from state
            nationality_key = current_labels.get(
                "data_point_legend_nationality", "nationality"
            )
            attribute_key = current_labels.get(
                "data_point_legend_attribute", "attribute"
            )
            color_map = {nationality_key: "red", attribute_key: "green"}

            print(f"[on_skip] Using keys: Nat='{nationality_key}', Attr='{attribute_key}'") # DEBUG PRINT
            print(f"[on_skip] New English Identity: {new_identity}, Display Identity: {new_identity_display}") # DEBUG PRINT

            return (
                # Update displayed value with translated identity and dynamic keys
                gr.update(
                    value=[(new_identity_display, nationality_key),
                           (new_attribute, attribute_key)],
                    color_map=color_map
                ),
                # Update the state holding the English data point
                [new_identity, new_attribute],
                None,  # Clear likert
                [],  # Clear nationalities dropdown
                "",
                gr.update(label=new_attr_label),
                gr.update(label=new_nat_label),
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
                language_labels_state,
                current_data_point_state
            ],
            outputs=[
                data_point_box,
                current_data_point_state,
                stereotype_likert,
                associated_nationalities_dropdown,
                associated_region_dropdown,
                associated_attributes_input,
                associated_attributes_input,
                associated_nationalities_dropdown,
            ],
        )

        skip_button.click(
            on_skip,
            inputs=[
                token_id,
                current_data_point_state,
                nationality_personal_info,
                language_labels_state,
            ],
            outputs=[
                data_point_box,
                current_data_point_state,
                stereotype_likert,
                associated_nationalities_dropdown,
                associated_attributes_input,
                associated_attributes_input,
                associated_nationalities_dropdown,
            ],
        )

        def toggle_chat(
            token_id,
            age,
            gender,
            nationality_personal_info,
            consent_checkbox,
            current_labels,
        ):
            print(f"[toggle_chat] Received current_labels from state: {current_labels}") # DEBUG PRINT
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
            )

            if is_valid:
                # Get a personalized data point using the user's information
                new_identity, new_attribute = get_random_data_point(
                    token_id=token_id,
                    nationality_personal_info=nationality_personal_info,
                )

                # Translate the new identity for display
                current_lang = current_labels.get("current_lang", "en")
                new_identity_display = translated_nationalities[current_lang].get(new_identity, new_identity)

                # Update the input labels with new data point values (using translated identity)
                new_attr_label, new_nat_label = update_input_labels(
                    new_identity_display, new_attribute, current_labels # Pass current_labels and translated identity
                )

                # Get current legend keys from state
                nationality_key = current_labels.get(
                    "data_point_legend_nationality", "nationality"
                )
                attribute_key = current_labels.get(
                    "data_point_legend_attribute", "attribute"
                )

                color_map = {nationality_key: "red", attribute_key: "green"}

                print(f"[toggle_chat] Using keys: Nat='{nationality_key}', Attr='{attribute_key}'") # DEBUG PRINT

                # Return updated UI state and the new data point
                return (
                    gr.Column(visible=True),
                    gr.Column(visible=False),
                    # Update displayed value with translated identity and dynamic keys
                    gr.update(
                        value=[(new_identity_display, nationality_key),
                               (new_attribute, attribute_key)],
                        color_map=color_map,
                    ),
                    # Update the state holding the English data point
                    [new_identity, new_attribute],
                    gr.update(label=new_attr_label),
                    gr.update(label=new_nat_label),
                )
            else:
                # Return original UI state without changing data point, keep existing data point value
                return (
                    gr.Column(visible=False),
                    gr.Column(visible=True),
                    gr.update(),  # Keep current data_point_box value
                    gr.update(),  # Keep current data point state
                    gr.update(),  # Keep current associated_attributes_input label
                    gr.update(),  # Keep current associated_nationalities_dropdown label
                )

        # Update the change event connections to include components in outputs (not their labels)
        # The outputs update the component values/visibility, the labels are updated via gr.update() within toggle_chat
        token_id.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                age,
                gender,
                nationality_personal_info,
                consent_checkbox,
                language_labels_state,
            ],
            outputs=[
                validator_col,
                personal_data_missing,
                data_point_box,
                current_data_point_state,
                associated_attributes_input,
                associated_nationalities_dropdown,
            ],
        )
        age.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                age,
                gender,
                nationality_personal_info,
                consent_checkbox,
                language_labels_state,
            ],
            outputs=[
                validator_col,
                personal_data_missing,
                data_point_box,
                current_data_point_state,
                associated_attributes_input,
                associated_nationalities_dropdown,
            ],
        )
        gender.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                age,
                gender,
                nationality_personal_info,
                consent_checkbox,
                language_labels_state,
            ],
            outputs=[
                validator_col,
                personal_data_missing,
                data_point_box,
                current_data_point_state,
                associated_attributes_input,
                associated_nationalities_dropdown,
            ],
        )
        nationality_personal_info.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                age,
                gender,
                nationality_personal_info,
                consent_checkbox,
                language_labels_state,
            ],
            outputs=[
                validator_col,
                personal_data_missing,
                data_point_box,
                current_data_point_state,
                associated_attributes_input,
                associated_nationalities_dropdown,
            ],
        )
        consent_checkbox.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                age,
                gender,
                nationality_personal_info,
                consent_checkbox,
                language_labels_state,
            ],
            outputs=[
                validator_col,
                personal_data_missing,
                data_point_box,
                current_data_point_state,
                associated_attributes_input,
                associated_nationalities_dropdown,
            ],
        )

        def toggle_and_update_regions(associated_nationalities_dropdown):
            if (
                associated_nationalities_dropdown is None
                or len(associated_nationalities_dropdown) == 0
            ):
                associated_region_dropdown = gr.Dropdown(
                    label=labels["associated_region_label"],
                    choices=[],
                    interactive=False, # Ensure it's disabled
                    multiselect=True,
                )
                # Return only the update for the dropdown itself
                return gr.update(choices=[], interactive=False)
            else:
                # Define helper inside or ensure it's accessible
                def get_administrative_divisions(selected_countries):
                    df = pd.read_json(
                        "data/global_administrative_division.json")
                    filtered_df = df[df["name"].isin(selected_countries)]
                    return [
                        f"{division['name']} ({row['name']})"
                        for _, row in filtered_df.iterrows()
                        for division in row["AD"]
                    ]

                associated_region_dropdown = gr.Dropdown(
                    label=labels["associated_region_label"],
                    choices=get_administrative_divisions(
                        associated_nationalities_dropdown
                    ),
                    multiselect=True,
                    interactive=True, # Enable interaction
                )
                # Return only the update for the dropdown itself
                return gr.update(choices=get_administrative_divisions(associated_nationalities_dropdown), interactive=True)

        associated_nationalities_dropdown.change(
            fn=toggle_and_update_regions,
            inputs=[associated_nationalities_dropdown],
            # Output only targets the region dropdown now
            outputs=[associated_region_dropdown],
        )

        # Language change handler function
        def on_language_change(lang_code, current_data_point): # Changed data_point input to current_data_point state
            # Load new labels for the selected language
            new_labels = load_language(lang_code)

            # Get the current English data point identity and attribute from state
            current_identity = current_data_point[0] if current_data_point else initial_identity
            current_attribute = current_data_point[1] if current_data_point and len(current_data_point) >= 2 \
                else initial_attribute # Corrected parenthesis placement

            # Get new translated legend keys
            new_nationality_key = new_labels.get(
                "data_point_legend_nationality", "nationality"
            )
            new_attribute_key = new_labels.get(
                "data_point_legend_attribute", "attribute"
            )
            new_dynamic_color_map = {
                new_nationality_key: "red",
                new_attribute_key: "green",
            }

            # Translate the current English identity using the new language code
            current_identity_display = translated_nationalities[lang_code].get(current_identity, current_identity)

            # Update dynamic labels (using translated identity for prompt format)
            new_attr_label, new_nat_label = update_input_labels(
                current_identity_display, current_attribute, new_labels # Pass new_labels and translated identity
            )

            print(f"[on_language_change] New labels for state: {new_labels}") # DEBUG PRINT
            print(f"[on_language_change] New keys: Nat='{new_nationality_key}', Attr='{new_attribute_key}'") # DEBUG PRINT

            # Get new nationality choices for the dropdowns
            new_nationality_choices = get_nationality_choices(lang_code)

            # Update all UI components with new language
            return (
                # State update
                new_labels,  # Output new labels to state
                # Personal info section
                gr.update(
                    label=new_labels["identifier_label"],
                    info=new_labels["identifier_info"],
                ),
                gr.update(label=new_labels["age_label"]),
                gr.update(label=new_labels["gender_label"]),
                gr.update(
                    label=new_labels["nationality_label"],
                    info=new_labels["nationality_info"],
                    # Update choices for the personal info nationality dropdown
                    choices=new_nationality_choices,
                ),
                gr.update(label=new_labels["consent_label"]),
                f"<a href='https://docs.google.com/document/d/1YEi0QpFYJwFBSIAjGplPc0VkOxJwnME29dWWyfp37XY/edit?usp=sharing'>{new_labels['consent_link_text']}</a>",
                new_labels["personal_data_missing_md"],
                # Main interface section
                new_labels["welcome_md"],
                # Data point display
                gr.update(
                    label=new_labels["data_point_label"],
                    value=[
                        (current_identity_display, new_nationality_key), # Use translated identity
                        (current_attribute, new_attribute_key),
                    ],
                    color_map=new_dynamic_color_map,
                ),
                # Stereotype section
                gr.update(
                    label=new_labels["likert_label"], info=new_labels["likert_info"]
                ),
                # Associated attributes and nationalities
                gr.update(
                    label=new_attr_label,
                    placeholder=new_labels["associated_attributes_placeholder"],
                ),
                # Update choices for the associated nationalities dropdown
                gr.update(label=new_nat_label, choices=new_nationality_choices),
                gr.update(label=new_labels["associated_region_label"]),
                # Buttons
                gr.update(value=new_labels["skip_button_label"]),
                gr.update(value=new_labels["submit_button_label"]),
            )

        # Connect language dropdown change handler to update the UI with the new language
        language_dropdown.change(
            fn=on_language_change,
            inputs=[language_dropdown, current_data_point_state], # Use state instead of data_point_box
            outputs=[
                # State
                language_labels_state,
                # Personal info
                token_id,
                age,
                gender,
                nationality_personal_info,
                consent_checkbox,
                consent_link_html,  # Updated to use the named variable
                personal_data_missing_md,  # Updated to use the named markdown component
                # Main interface section
                welcome_md,  # Updated to use the named markdown component
                # Data display
                data_point_box,
                # Form elements
                stereotype_likert,
                associated_attributes_input,
                associated_nationalities_dropdown,
                associated_region_dropdown,
                # Buttons
                skip_button,
                submit_button,
            ],
        )

    return interface
