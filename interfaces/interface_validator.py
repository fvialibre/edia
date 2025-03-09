import json
import os
from datetime import datetime

import country_converter as coco
import gradio as gr
import pandas as pd
from data.nationalities import nationalities

from interfaces.data_selection import select_data_point


# --- Interface ---
def interface() -> gr.Blocks:
    # Set up country converter
    coco.logging.getLogger().setLevel(coco.logging.CRITICAL)
    cc = coco.CountryConverter(only_UNmember=True)

    # Check for required files
    required_files = {
        "data/processed_Frases_HESEIA_Anotación.csv": "HESEIA dataset",
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
        pd.DataFrame(columns=["identity", "attribute", "skip_count"]).to_csv(
            skip_csv_path, index=False
        )

    # Load required datasets
    df_heseia = pd.read_csv("data/processed_Frases_HESEIA_Anotación.csv")
    df_borders = pd.read_csv("data/country_borders.csv")

    df_heseia = df_heseia[df_heseia["region_type"] == "País"]
    df_heseia = df_heseia[
        df_heseia[
            "Expresa un estereotipo que conocen?\n1 (muy en desacuerdo) - 5 (muy de acuerdo)"
        ]
        >= 4
    ]

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

        # Update the CSV file with skip counts
        if os.path.exists(skip_csv_path):
            df_skips = pd.read_csv(skip_csv_path)
        else:
            df_skips = pd.DataFrame(columns=["identity", "attribute", "skip_count"])

        # Look for existing entry
        mask = (df_skips["identity"] == identity) & (df_skips["attribute"] == attribute)
        if mask.any():
            # Increment skip_count for existing entry
            df_skips.loc[mask, "skip_count"] += 1
        else:
            # Add new entry with skip_count = 1
            new_row = pd.DataFrame(
                {"identity": [identity], "attribute": [attribute], "skip_count": [1]}
            )
            df_skips = pd.concat([df_skips, new_row], ignore_index=True)

        # Save updated skip counts
        df_skips.to_csv(skip_csv_path, index=False)

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
            # Read existing stereotypes
            df_ws_stereotypes = pd.read_csv(ws_stereotypes_path)

            # Append new stereotypes
            df_ws_stereotypes = pd.concat(
                [df_ws_stereotypes, pd.DataFrame(new_stereotypes)], ignore_index=True
            )

            # Save back to file
            df_ws_stereotypes.to_csv(ws_stereotypes_path, index=False)

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
            f"Which other attributes do you associate with {identity}?",
            f"Which other nationalities do you associate with '{attribute}'?",
        )

    # Get initial labels
    initial_attr_label, initial_nat_label = update_input_labels(initial_identity, initial_attribute)

    # Gradio interface
    with gr.Blocks() as interface:
        with gr.Row():
            token_id = gr.Textbox(
                label="Identifier",
                info="Enter the identifier provided in the workshop",
                lines=1,
            )
            age = gr.Number(
                value=0,
                label="Enter your age",
                visible=False,
            )
            gender = gr.Radio(
                ["M", "F", "X"],
                label="Select your gender",
                value="X",
                visible=False,
            )
            nationality_personal_info = gr.Dropdown(
                label="Where are you from?",
                info="Select the nationality that represents your cultural, personal, or national identity",
                choices=nationalities,
                multiselect=True,
                allow_custom_value=False,
            )
            with gr.Column():
                consent_checkbox = gr.Checkbox(
                    label="I have read and accept the informed consent ⬇️", value=False
                )
                _ = gr.HTML(
                    value="<a href='https://docs.google.com/document/d/1YEi0QpFYJwFBSIAjGplPc0VkOxJwnME29dWWyfp37XY/edit?usp=sharing'>Link 🔗</a>",
                )
        _ = gr.HTML(
            value="<hr>",
        )
        with gr.Column(visible=True) as personal_data_missing:
            gr.Markdown(
                """
                # Enter your personal data and confirm your consent to proceed with the survey!
            """
            )
        with gr.Column(visible=False, elem_id="col") as validator_col:
            _ = gr.Markdown(
                """
                # Welcome to the Stereotype Validator

                ### This tool is designed to help us understand how stereotypes are perceived in different regions.

                ### Each time you submit a response, you will be presented with a new data point from the dataset.
                """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    data_point_box = gr.HighlightedText(
                        label="Random data point",
                        value=[
                            (initial_identity, "nationality"),
                            (initial_attribute, "attribute"),
                        ],
                        combine_adjacent=True,
                        show_legend=True,
                        interactive=False,
                        color_map={"nationality": "red", "attribute": "green"},
                    )
                with gr.Column(scale=1):
                    stereotype_likert = gr.Radio(
                        [1, 2, 3, 4, 5],
                        label="This is a known association in my region",
                        info="1: Strongly disagree, 5: Strongly agree",
                        interactive=True,
                    )
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    associated_attributes_input = gr.Textbox(
                        label=initial_attr_label,
                        placeholder="Enter attributes separated by commas",
                    )
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    associated_nationalities_dropdown = gr.Dropdown(
                        label=initial_nat_label,
                        choices=nationalities,
                        multiselect=True,
                    )
                with gr.Column(
                    visible=False, scale=1
                ) as associated_region_dropdown_col:
                    associated_region_dropdown = gr.Dropdown(
                        label="Any specific region?",
                        choices=nationalities,
                        multiselect=True,
                    )
            with gr.Row(equal_height=True):
                skip_button = gr.Button("Skip", variant="primary", scale=25)
                submit_button = gr.Button("Submit", variant="secondary", scale=75)

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

            return (
                [(new_identity, "nationality"), (new_attribute, "attribute")],
                None,
                [],
                [],
                "",
                gr.update(label=new_attr_label),
                gr.update(label=new_nat_label),
            )

        def on_skip(token_id, data_point, nationality_personal_info):
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

            return (
                [(new_identity, "nationality"), (new_attribute, "attribute")],
                None,
                [],
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
            token_id, age, gender, nationality_personal_info, consent_checkbox
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

                # Return updated UI state and the new data point
                return (
                    gr.Column(visible=True),
                    gr.Column(visible=False),
                    [(new_identity, "nationality"), (new_attribute, "attribute")],
                    gr.update(label=new_attr_label),
                    gr.update(label=new_nat_label),
                )
            else:
                # Return original UI state without changing data point
                return (
                    gr.Column(visible=False),
                    gr.Column(visible=True),
                    None,
                    gr.update(),  # Keep current label
                    gr.update(),  # Keep current label
                )

        # Update the change event connections to include components in outputs (not their labels)
        token_id.change(
            fn=toggle_chat,
            inputs=[token_id, age, gender, nationality_personal_info, consent_checkbox],
            outputs=[validator_col, personal_data_missing, data_point_box,
                    associated_attributes_input, associated_nationalities_dropdown],
        )
        age.change(
            fn=toggle_chat,
            inputs=[token_id, age, gender, nationality_personal_info, consent_checkbox],
            outputs=[validator_col, personal_data_missing, data_point_box,
                    associated_attributes_input, associated_nationalities_dropdown],
        )
        gender.change(
            fn=toggle_chat,
            inputs=[token_id, age, gender, nationality_personal_info, consent_checkbox],
            outputs=[validator_col, personal_data_missing, data_point_box,
                    associated_attributes_input, associated_nationalities_dropdown],
        )
        nationality_personal_info.change(
            fn=toggle_chat,
            inputs=[token_id, age, gender, nationality_personal_info, consent_checkbox],
            outputs=[validator_col, personal_data_missing, data_point_box,
                    associated_attributes_input, associated_nationalities_dropdown],
        )
        consent_checkbox.change(
            fn=toggle_chat,
            inputs=[token_id, age, gender, nationality_personal_info, consent_checkbox],
            outputs=[validator_col, personal_data_missing, data_point_box,
                    associated_attributes_input, associated_nationalities_dropdown],
        )

        def toggle_and_update_regions(associated_nationalities_dropdown):
            if (
                associated_nationalities_dropdown is None
                or len(associated_nationalities_dropdown) == 0
            ):
                associated_region_dropdown = gr.Dropdown(
                    label="Any specific region?", choices=[], multiselect=True
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
                    label="Any specific region?",
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

    return interface
