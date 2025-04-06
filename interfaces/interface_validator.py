import json
import os
from datetime import datetime

import country_converter as coco
import gradio as gr
import pandas as pd
from data.nationalities import nationalities

from interfaces.data_selection import select_data_point

# --- Language Handling ---
AVAILABLE_LANGUAGES = {"English": "en", "Español": "es", "Português": "pt"}
DEFAULT_LANG = "es"  # Default starting language

def load_language(lang: str):
    """Loads language labels for the validator interface."""
    labels_path = f"language/{lang}.json"
    fallback_path = f"language/en.json"  # English as fallback

    if not os.path.exists(labels_path):
        print(f"Warning: Language file {labels_path} not found. Defaulting to English.")
        labels_path = fallback_path
        lang = "en"  # Update lang if falling back

    try:
        # Using pandas consistent with interface_crowsPairs.py
        all_labels = pd.read_json(labels_path)
        # Use a key consistent with others, e.g., "validator_interface"
        labels = all_labels["validator_interface"]
        # Add the current language code to the labels dict for reference
        labels["current_lang"] = lang
        return labels
    except KeyError:
        # Handle missing key - maybe load English as fallback?
        print(f"Warning: 'validator_interface' key not found in {labels_path}. Loading English.")
        all_labels = pd.read_json(fallback_path)
        try:
            labels = all_labels["validator_interface"]  # Assuming English file has the key
            labels["current_lang"] = "en"  # Mark as fallback lang
            return labels
        except KeyError:
             raise RuntimeError(f"Critical: 'validator_interface' key not found in fallback English file {fallback_path}")
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
            columns=["identity", "attribute", "annotator_id", "annotator_nationalities"]
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
            [{"identity": identity, "attribute": attribute, "annotator_id": annotator_id}]
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
            attributes_list = [
                attr.strip()
                for attr in associated_attributes.split(",")
                if attr.strip()
            ]
            for attr in attributes_list:
                new_stereotypes.append(
                    {
                        "identity": identity,
                        "attribute": attr,
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
    def update_input_labels(identity, attribute):
        return (
            labels['associated_attributes_prompt'].format(identity=identity),
            labels['associated_nationalities_prompt'].format(attribute=attribute),
        )

    # Get initial labels (will be set later in the Gradio Blocks definition)
    # initial_attr_label, initial_nat_label = update_input_labels(initial_identity, initial_attribute)

    # Get translated keys for HighlightedText
    nationality_key = labels.get('data_point_legend_nationality', 'nationality')
    attribute_key = labels.get('data_point_legend_attribute', 'attribute')
    dynamic_color_map = {nationality_key: "red", attribute_key: "green"}

    # Gradio interface
    # Get initial labels for dynamic fields before building the UI
    initial_attr_label, initial_nat_label = update_input_labels(initial_identity, initial_attribute)

    with gr.Blocks() as interface:
        # State to hold the current language labels
        language_labels_state = gr.State(labels)

        # Language selector at the top of the interface
        with gr.Row():
            language_dropdown = gr.Dropdown(
                label="Language / Idioma / Idioma", # Multilingual label
                choices=list(AVAILABLE_LANGUAGES.items()),
                value=lang, # Default to server-provided language
                interactive=True,
                elem_id="language_dropdown"
            )
            gr.HTML("<div style='flex-grow: 1'></div>") # Spacer

        # Personal information row
        with gr.Row():
            token_id = gr.Textbox(
                label=labels['identifier_label'],
                info=labels['identifier_info'],
                lines=1,
            )
            age = gr.Number(
                value=0,
                label=labels['age_label'],
                visible=False,
            )
            gender = gr.Radio(
                # Assuming M/F/X are universal codes, otherwise these need translation too
                ["M", "F", "X"],
                label=labels['gender_label'],
                value="X",
                visible=False,
            )
            nationality_personal_info = gr.Dropdown(
                label=labels['nationality_label'],
                info=labels['nationality_info'],
                choices=nationalities, # Keep nationalities list as is, assuming it's language-independent data
                multiselect=True,
                allow_custom_value=False,
            )
            with gr.Column():
                consent_checkbox = gr.Checkbox(
                    label=labels['consent_label'], value=False
                )
                consent_link_html = gr.HTML(
                    value=f"<a href='https://docs.google.com/document/d/1YEi0QpFYJwFBSIAjGplPc0VkOxJwnME29dWWyfp37XY/edit?usp=sharing'>{labels['consent_link_text']}</a>",
                    elem_id="consent_link_html"
                )

        gr.HTML("<hr>")

        # Create Markdown components within their proper context
        with gr.Column(visible=True) as personal_data_missing:
            personal_data_missing_md = gr.Markdown(labels['personal_data_missing_md'])
        with gr.Column(visible=False, elem_id="col") as validator_col:
            welcome_md = gr.Markdown(labels['welcome_md'])
            with gr.Row():
                with gr.Column(scale=1):
                    # Assuming color_map keys 'nationality' and 'attribute' are internal identifiers
                    # If the displayed legend text needs translation, we'd need more complex setup
                    # For now, let's assume the legend text comes from the tuple values directly
                    # and the label needs translation.
                    data_point_box = gr.HighlightedText(
                        label=labels['data_point_label'],
                        value=[
                            (initial_identity, nationality_key), # Use dynamic key
                            (initial_attribute, attribute_key), # Use dynamic key
                        ],
                        combine_adjacent=True,
                        show_legend=True,
                        interactive=False,
                        color_map=dynamic_color_map, # Use dynamic map
                    )
                with gr.Column(scale=1):
                    stereotype_likert = gr.Radio(
                        [1, 2, 3, 4, 5], # Assuming numbers are universal
                        label=labels['likert_label'],
                        info=labels['likert_info'],
                        interactive=True,
                    )
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    associated_attributes_input = gr.Textbox(
                        label=initial_attr_label, # Already dynamically set
                        placeholder=labels['associated_attributes_placeholder'],
                    )
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    associated_nationalities_dropdown = gr.Dropdown(
                        label=initial_nat_label, # Already dynamically set
                        choices=nationalities,
                        multiselect=True,
                    )
                with gr.Column(
                    visible=False, scale=1
                ) as associated_region_dropdown_col:
                    associated_region_dropdown = gr.Dropdown(
                        label=labels['associated_region_label'],
                        choices=nationalities,
                        multiselect=True,
                    )
            with gr.Row(equal_height=True):
                skip_button = gr.Button(labels['skip_button_label'], variant="primary", scale=25)
                submit_button = gr.Button(labels['submit_button_label'], variant="secondary", scale=75)

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
            current_labels, # Added state input
        ):

            log_result(
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
            )
            new_identity, new_attribute = get_random_data_point(
                token_id=token_id, nationality_personal_info=nationality_personal_info
            )

            # Update the input labels with new data point values
            new_attr_label, new_nat_label = update_input_labels(new_identity, new_attribute)

            # Get current legend keys from state
            nationality_key = current_labels.get('data_point_legend_nationality', 'nationality')
            attribute_key = current_labels.get('data_point_legend_attribute', 'attribute')

            return (
                # Update displayed value with dynamic keys from state
                [(new_identity, nationality_key),
                 (new_attribute, attribute_key)],
                None, # Clear likert
                [], # Clear nationalities dropdown
                [],
                "",
                gr.update(label=new_attr_label),
                gr.update(label=new_nat_label),
            )

        def on_skip(token_id, data_point, nationality_personal_info, current_labels): # Added state input
            # Extract current identity and attribute from data_point
            if data_point and len(data_point) >= 2:
                identity = data_point[0]["token"]
                attribute = data_point[1]["token"]

                # Log the skip if we have valid identity and attribute
                if identity and attribute:
                    log_skip(identity, attribute, token_id)

            # Get new data point, taking skip counts into consideration
            new_identity, new_attribute = get_random_data_point(
                token_id=token_id,
                nationality_personal_info=nationality_personal_info,
            )

            # Update the input labels with new data point values
            new_attr_label, new_nat_label = update_input_labels(new_identity, new_attribute)

            # Get current legend keys from state
            nationality_key = current_labels.get('data_point_legend_nationality', 'nationality')
            attribute_key = current_labels.get('data_point_legend_attribute', 'attribute')

            return (
                 # Update displayed value with dynamic keys from state
                [(new_identity, nationality_key),
                 (new_attribute, attribute_key)],
                None, # Clear likert
                [], # Clear nationalities dropdown
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
                language_labels_state, # Added state input
            ],
            outputs=[
                data_point_box,
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
                data_point_box,
                nationality_personal_info,  # Add nationality_personal_info as input
                language_labels_state, # Added state input
            ],
            outputs=[
                data_point_box,
                stereotype_likert,
                associated_nationalities_dropdown,
                associated_attributes_input,
                associated_attributes_input,
                associated_nationalities_dropdown,
            ],
        )

        def toggle_chat(
            token_id, age, gender, nationality_personal_info, consent_checkbox, current_labels # Added state input
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
            )

            if is_valid:
                # Get a personalized data point using the user's information
                new_identity, new_attribute = get_random_data_point(
                    token_id=token_id,
                    nationality_personal_info=nationality_personal_info
                )

                # Update the input labels with new data point values
                new_attr_label, new_nat_label = update_input_labels(new_identity, new_attribute)

                # Get current legend keys from state
                nationality_key = current_labels.get('data_point_legend_nationality', 'nationality')
                attribute_key = current_labels.get('data_point_legend_attribute', 'attribute')

                # Return updated UI state and the new data point
                return (
                    gr.Column(visible=True),
                    gr.Column(visible=False),
                     # Update displayed value with dynamic keys from state
                    [(new_identity, nationality_key),
                     (new_attribute, attribute_key)],
                    gr.update(label=new_attr_label),
                    gr.update(label=new_nat_label),
                )
            else:
                # Return original UI state without changing data point, keep existing data point value
                return (
                    gr.Column(visible=False),
                    gr.Column(visible=True),
                    gr.update(), # Keep current data_point_box value
                    gr.update(), # Keep current associated_attributes_input label
                    gr.update(), # Keep current associated_nationalities_dropdown label
                )

        # Update the change event connections to include components in outputs (not their labels)
        # The outputs update the component values/visibility, the labels are updated via gr.update() within toggle_chat
        token_id.change(
            fn=toggle_chat,
            inputs=[token_id, age, gender, nationality_personal_info, consent_checkbox, language_labels_state], # Added state input
            outputs=[validator_col, personal_data_missing, data_point_box,
                    associated_attributes_input, associated_nationalities_dropdown],
        )
        age.change(
            fn=toggle_chat,
            inputs=[token_id, age, gender, nationality_personal_info, consent_checkbox, language_labels_state], # Added state input
            outputs=[validator_col, personal_data_missing, data_point_box,
                    associated_attributes_input, associated_nationalities_dropdown],
        )
        gender.change(
            fn=toggle_chat,
            inputs=[token_id, age, gender, nationality_personal_info, consent_checkbox, language_labels_state], # Added state input
            outputs=[validator_col, personal_data_missing, data_point_box,
                    associated_attributes_input, associated_nationalities_dropdown],
        )
        nationality_personal_info.change(
            fn=toggle_chat,
            inputs=[token_id, age, gender, nationality_personal_info, consent_checkbox, language_labels_state], # Added state input
            outputs=[validator_col, personal_data_missing, data_point_box,
                    associated_attributes_input, associated_nationalities_dropdown],
        )
        consent_checkbox.change(
            fn=toggle_chat,
            inputs=[token_id, age, gender, nationality_personal_info, consent_checkbox, language_labels_state], # Added state input
            outputs=[validator_col, personal_data_missing, data_point_box,
                    associated_attributes_input, associated_nationalities_dropdown],
        )

        def toggle_and_update_regions(associated_nationalities_dropdown):
            if (
                associated_nationalities_dropdown is None
                or len(associated_nationalities_dropdown) == 0
            ):
                associated_region_dropdown = gr.Dropdown(
                    label=labels['associated_region_label'], choices=[], multiselect=True
                )
                return gr.Column(visible=False), associated_region_dropdown
            else:

                def get_administrative_divisions(selected_countries):
                    df = pd.read_json("data/global_administrative_division.json")
                    filtered_df = df[df["name"].isin(selected_countries)]
                    return [
                        f"{division['name']} ({row['name']})"
                        for _, row in filtered_df.iterrows()
                        for division in row["AD"]
                    ]

                associated_region_dropdown = gr.Dropdown(
                    label=labels['associated_region_label'],
                    choices=get_administrative_divisions(
                        associated_nationalities_dropdown
                    ),
                    multiselect=True,
                    interactive=True,
                )
                return gr.Column(visible=True), associated_region_dropdown

        associated_nationalities_dropdown.change(
            fn=toggle_and_update_regions,
            inputs=[associated_nationalities_dropdown],
            outputs=[associated_region_dropdown_col, associated_region_dropdown],
        )

        # Language change handler function
        def on_language_change(lang_code, data_point):
            # Load new labels for the selected language
            new_labels = load_language(lang_code)

            # Get the current data point identity and attribute if available
            current_identity = data_point[0]["token"] if data_point and len(data_point) >= 1 else initial_identity
            current_attribute = data_point[1]["token"] if data_point and len(data_point) >= 2 else initial_attribute

            # Get new translated legend keys
            new_nationality_key = new_labels.get('data_point_legend_nationality', 'nationality')
            new_attribute_key = new_labels.get('data_point_legend_attribute', 'attribute')
            new_dynamic_color_map = {new_nationality_key: "red", new_attribute_key: "green"}

            # Update dynamic labels
            new_attr_label, new_nat_label = update_input_labels(current_identity, current_attribute)

            # Update all UI components with new language
            return (
                # State update
                new_labels, # Output new labels to state

                # Personal info section
                gr.update(label=new_labels['identifier_label'], info=new_labels['identifier_info']),
                gr.update(label=new_labels['age_label']),
                gr.update(label=new_labels['gender_label']),
                gr.update(label=new_labels['nationality_label'], info=new_labels['nationality_info']),
                gr.update(label=new_labels['consent_label']),
                f"<a href='https://docs.google.com/document/d/1YEi0QpFYJwFBSIAjGplPc0VkOxJwnME29dWWyfp37XY/edit?usp=sharing'>{new_labels['consent_link_text']}</a>",
                new_labels['personal_data_missing_md'],

                # Main interface section
                new_labels['welcome_md'],

                # Data point display
                gr.update(
                    label=new_labels['data_point_label'],
                    value=[(current_identity, new_nationality_key), (current_attribute, new_attribute_key)],
                    color_map=new_dynamic_color_map
                ),

                # Stereotype section
                gr.update(label=new_labels['likert_label'], info=new_labels['likert_info']),

                # Associated attributes and nationalities
                gr.update(label=new_attr_label, placeholder=new_labels['associated_attributes_placeholder']),
                gr.update(label=new_nat_label),
                gr.update(label=new_labels['associated_region_label']),

                # Buttons
                gr.update(value=new_labels['skip_button_label']),
                gr.update(value=new_labels['submit_button_label'])
            )

        # Connect language dropdown change handler to update the UI with the new language
        language_dropdown.change(
            fn=on_language_change,
            inputs=[language_dropdown, data_point_box],
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
                submit_button
            ]
        )

    return interface
