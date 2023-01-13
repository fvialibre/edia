import pandas as pd
from typing import List

class ErrorManager:
    def __init__(
        self, 
        path: str,
        str_to_prepend: str='',
        str_to_append: str=''
    ) -> None:
        self.error2text = pd.read_json(path)["errors"]

        self.str_to_prepend = str_to_prepend
        self.str_to_append  = str_to_append

    def __get_text_from_code(
        self, 
        error_info: str
    ) -> str:
        error_code = error_info[0]
        error_args = error_info[1:]
        return str(self.error2text[error_code]).format(*error_args)

    def process(
        self, 
        error_info: List[str],
    ) -> str:
        if not error_info:
            return ""
        
        error = self.__get_text_from_code(error_info=error_info)
        
        return self.str_to_prepend + error + self.str_to_append