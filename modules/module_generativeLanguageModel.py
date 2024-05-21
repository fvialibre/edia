from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Disabling parallelism to avoid deadlocks in the hf tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class GenerativeLanguageModel:
    def __init__(
        self, 
        model_name
    ) -> None:
    
        print("Downloading generative language model...")

        self.__tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.__model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda:1",
        )
    
    def initTokenizer(
        self
    ) -> AutoTokenizer:

        return self.__tokenizer
    
    def initModel(
        self
    ) -> AutoModelForCausalLM:

        return self.__model
