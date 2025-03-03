import gradio as gr
import pandas as pd
import random
import json
from datetime import datetime
from data.nationalities import nationalities
from auth import school_list

# --- Interface ---
def interface() -> gr.Blocks:
    df = pd.read_csv('data/processed_Frases_HESEIA_Anotación.csv')
    df = df[df['region_type'] == 'País']
    df = df[df['Expresa un estereotipo que conocen?\n1 (muy en desacuerdo) - 5 (muy de acuerdo)'] >= 4]

    def get_random_data_point():
        random_row = df.sample(1).iloc[0]
        return random_row['region'], random_row['attribute']

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
        associated_attributes
    ):
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
        with open('logs/logs_validator.jsonl', 'a+', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    initial_identity, initial_attribute = get_random_data_point()

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
                    label='I have read and accept the informed consent ⬇️',
                    value=False
                )
                _ = gr.HTML(
                    value="<a href='https://docs.google.com/document/d/17Feum83dTqjcicgJxuWdZ3qLuL3emmVY2idGym_usLU/edit?usp=sharing'>Link 🔗</a>",
                )
        _ = gr.HTML(
            value="<hr>",
        )
        with gr.Column(visible=True) as personal_data_missing:
            gr.Markdown("""
                # Enter your personal data and confirm your consent to proceed with the survey!
            """)
        with gr.Column(visible=False, elem_id='col') as validator_col:
            _ = gr.Markdown(
                """
                # Welcome to the Stereotype Validator
                
                ### This tool is designed to help us understand how stereotypes are perceived in different regions.
                
                ### Each time you submit a response, you will be presented with a new data point from the dataset.
                """
            )
            with gr.Row():
                data_point_box = gr.HighlightedText(
                    label="Random data point",
                    value=[(initial_identity, "nationality"), (initial_attribute, "attribute")],
                    combine_adjacent=True,
                    show_legend=True,
                    interactive=False,
                    color_map={"nationality": "red", "attribute": "green"}
                )
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    associated_attributes_input = gr.Textbox(
                        label="Which other attributes do you associate with this nationality?",
                        placeholder="Enter attributes separated by commas"
                    )
                with gr.Column(scale=1):
                    stereotype_likert = gr.Radio(
                        [1,2,3,4,5],
                        label="This is a known stereotype in my region",
                        info="1: Strongly disagree, 5: Strongly agree",
                        interactive=True,
                    )
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    associated_nationalities_dropdown = gr.Dropdown(
                        label="Which other nationalities do you associate with this attribute?",
                        choices=nationalities,
                        multiselect=True
                    )
                with gr.Column(visible=False, scale=1) as associated_region_dropdown_col:
                    associated_region_dropdown = gr.Dropdown(
                        label="Any specific region?",
                        choices=nationalities,
                        multiselect=True
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
                associated_attributes
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
                associated_attributes
            )
            new_identity, new_attribute = get_random_data_point()
            return [(new_identity, "nationality"), (new_attribute, "attribute")], None, [], [], ""

        def on_skip():
            new_identity, new_attribute = get_random_data_point()
            return [(new_identity, "nationality"), (new_attribute, "attribute")], None, [], ""

        submit_button.click(on_submit,
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
                associated_attributes_input
            ],
            outputs=[
                data_point_box,
                stereotype_likert,
                associated_nationalities_dropdown,
                associated_region_dropdown,
                associated_attributes_input
            ]
        )

        skip_button.click(on_skip,
            outputs=[
                data_point_box,
                stereotype_likert,
                associated_nationalities_dropdown,
                associated_attributes_input
            ]
        )

        def toggle_chat(token_id, age, gender, nationality_personal_info, consent_checkbox):
            if not (token_id is None
                or age is None
                or gender is None
                or nationality_personal_info is None
                or consent_checkbox is None
                or age < 0
                or age > 100
                or len(nationality_personal_info) == 0
                or len(token_id) == 0
                or not consent_checkbox):
                return gr.Column(visible=True), gr.Column(visible=False)
            else:
                return gr.Column(visible=False), gr.Column(visible=True)

        token_id.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                age,
                gender,
                nationality_personal_info,
                consent_checkbox
            ],
            outputs=[validator_col, personal_data_missing])
        age.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                age,
                gender,
                nationality_personal_info,
                consent_checkbox
            ],
            outputs=[validator_col, personal_data_missing])
        gender.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                age,
                gender,
                nationality_personal_info,
                consent_checkbox
            ],
            outputs=[validator_col, personal_data_missing])
        nationality_personal_info.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                age,
                gender,
                nationality_personal_info,
                consent_checkbox
            ],
            outputs=[validator_col, personal_data_missing])
        consent_checkbox.change(
            fn=toggle_chat,
            inputs=[
                token_id,
                age,
                gender,
                nationality_personal_info,
                consent_checkbox
            ],
            outputs=[validator_col, personal_data_missing])

        def toggle_and_update_regions(associated_nationalities_dropdown):
            if (associated_nationalities_dropdown is None
                or len(associated_nationalities_dropdown) == 0):
                associated_region_dropdown = gr.Dropdown(
                    label="Any specific region?",
                    choices=[],
                    multiselect=True
                )
                return gr.Column(visible=False), associated_region_dropdown
            else:
                def get_administrative_divisions(selected_countries):
                    df = pd.read_json('data/global_administrative_division.json')
                    filtered_df = df[df['name'].isin(selected_countries)]
                    return [f"{division['name']} ({row['name']})" for _, row in filtered_df.iterrows() for division in row['AD']]

                associated_region_dropdown = gr.Dropdown(
                    label="Any specific region?",
                    choices=get_administrative_divisions(associated_nationalities_dropdown),
                    multiselect=True,
                    interactive=True,
                )
                return gr.Column(visible=True), associated_region_dropdown

        associated_nationalities_dropdown.change(
            fn=toggle_and_update_regions,
            inputs=[
                associated_nationalities_dropdown
            ],
            outputs=[associated_region_dropdown_col, associated_region_dropdown])
                
    return interface
