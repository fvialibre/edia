from distutils.log import debug
from gradio.flagging import FlaggingCallback, _get_dataset_features_info
from gradio.components import IOComponent
from gradio import utils
from typing import Any, List, Optional
from dotenv import load_dotenv
from datetime import datetime
import csv, os, pytz


# --- Load environments vars ---
load_dotenv()


# --- Classes declaration ---
class DateLogs:
    def __init__(self, zone="America/Argentina/Cordoba"):
        self.time_zone = pytz.timezone(zone)
        
    def full(self):
        now = datetime.now(self.time_zone)
        return now.strftime("%H:%M:%S %d-%m-%Y")
    
    def day(self):
        now = datetime.now(self.time_zone)
        return now.strftime("%d-%m-%Y")

class HuggingFaceDatasetSaver(FlaggingCallback):
    """
    A callback that saves each flagged sample (both the input and output data)
    to a HuggingFace dataset.
    Example:
        import gradio as gr
        hf_writer = gr.HuggingFaceDatasetSaver(HF_API_TOKEN, "image-classification-mistakes")
        def image_classifier(inp):
            return {'cat': 0.3, 'dog': 0.7}
        demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label",
                            allow_flagging="manual", flagging_callback=hf_writer)
    Guides: using_flagging
    """

    def __init__(
        self,
        hf_token: str = os.getenv('HF_TOKEN'),
        dataset_name: str = os.getenv('DS_LOGS_NAME'),
        organization: Optional[str] = os.getenv('ORG_NAME'),
        private: bool = True,
        available_logs: bool = False
    ):
        """
        Parameters:
            hf_token: The HuggingFace token to use to create (and write the flagged sample to) the HuggingFace dataset.
            dataset_name: The name of the dataset to save the data to, e.g. "image-classifier-1"
            organization: The organization to save the dataset under. The hf_token must provide write access to this organization. If not provided, saved under the name of the user corresponding to the hf_token.
            private: Whether the dataset should be private (defaults to False).
        """
        self.hf_token = hf_token
        self.dataset_name = dataset_name
        self.organization_name = organization
        self.dataset_private = private
        self.datetime = DateLogs()
        self.available_logs = available_logs

        if not available_logs:
            print("Push: logs DISABLED!...")
        

    def setup(
            self, 
            components: List[IOComponent],
            flagging_dir: str
        ):
        """
        Params:
        flagging_dir (str): local directory where the dataset is cloned,
        updated, and pushed from.
        """
        if self.available_logs:
            
            try:
                import huggingface_hub
            except (ImportError, ModuleNotFoundError):
                raise ImportError(
                    "Package `huggingface_hub` not found is needed "
                    "for HuggingFaceDatasetSaver. Try 'pip install huggingface_hub'."
                )

            path_to_dataset_repo = huggingface_hub.create_repo(
                repo_id=os.path.join(self.organization_name, self.dataset_name),
                token=self.hf_token,
                private=self.dataset_private,
                repo_type="dataset",
                exist_ok=True,
            )

            self.path_to_dataset_repo = path_to_dataset_repo
            self.components = components
            self.flagging_dir = flagging_dir
            self.dataset_dir = self.dataset_name

            self.repo = huggingface_hub.Repository(
                local_dir=self.dataset_dir,
                clone_from=path_to_dataset_repo,
                use_auth_token=self.hf_token,
            )
            
            self.repo.git_pull(lfs=True)

            # Should filename be user-specified?
            # log_file_name = self.datetime.day()+"_"+self.flagging_dir+".csv"
            self.log_file = os.path.join(self.dataset_dir, self.flagging_dir+".csv")

    def flag(
        self,
        flag_data: List[Any],
        flag_option: Optional[str] = None,
        flag_index: Optional[int] = None,
        username: Optional[str] = None,
    ) -> int:

        if self.available_logs:
            self.repo.git_pull(lfs=True)

            is_new = not os.path.exists(self.log_file)

            with open(self.log_file, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)

                # File previews for certain input and output types
                infos, file_preview_types, headers = _get_dataset_features_info(
                    is_new, self.components
                )

                # Generate the headers and dataset_infos
                if is_new:
                    headers = [
                        component.label or f"component {idx}"
                        for idx, component in enumerate(self.components)
                    ] + [
                        "flag",
                        "username",
                        "timestamp",
                    ]
                    writer.writerow(utils.sanitize_list_for_csv(headers))

                # Generate the row corresponding to the flagged sample
                csv_data = []
                for component, sample in zip(self.components, flag_data):
                    save_dir = os.path.join(
                        self.dataset_dir,
                        utils.strip_invalid_filename_characters(component.label),
                    )
                    filepath = component.deserialize(sample, save_dir, None)
                    csv_data.append(filepath)
                    if isinstance(component, tuple(file_preview_types)):
                        csv_data.append(
                            "{}/resolve/main/{}".format(self.path_to_dataset_repo, filepath)
                        )

                csv_data.append(flag_option if flag_option is not None else "")
                csv_data.append(username if username is not None else "")
                csv_data.append(self.datetime.full())
                writer.writerow(utils.sanitize_list_for_csv(csv_data))


            with open(self.log_file, "r", encoding="utf-8") as csvfile:
                line_count = len([None for row in csv.reader(csvfile)]) - 1

            self.repo.push_to_hub(commit_message="Flagged sample #{}".format(line_count))
        
        else:
            line_count = 0
            print("Logs: Virtual push...")
            
        return line_count