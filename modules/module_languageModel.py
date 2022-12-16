from transformers import AutoTokenizer, AutoModelForMaskedLM

class LanguageModel:
    def __init__(
        self, 
        model_name
    ) -> None:
    
        print("Downloading language model...")
        self.__tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.__model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    def initTokenizer(
        self
    ) -> AutoTokenizer:

        return self.__tokenizer
    
    def initModel(
        self
    ) -> AutoModelForMaskedLM:

        return self.__model
