from typing import List, Dict

class CustomPllLabel:
    def __init__(
        self
    ) -> None:

        self.html_head = """
        <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    progress {
                        -webkit-appearance: none;
                    }
                    progress::-webkit-progress-value {
                        background-color: #4B3887;
                        border-radius: 0.5em;
                    }
                    progress::-webkit-progress-bar {
                        background-color: white;
                        border-radius: 0.5em;
                    }
                    #myturn span {
                        position: absolute;
                        display: inline-block;
                        color: #fff;
                        text-align: right;
                        font-size:15px
                    }
                    #myturn {
                        display: block;
                        position: relative;
                        margin: auto;
                        width: 90%;
                        padding: 2px;
                    }
                    progress {
                        width:100%;
                        height:20px;
                        border-radius: 7px;
                    }
                </style>
            </head>
            <body>
        """

        self.html_footer ="</body></html>"
    
    def __progressbar(
        self, 
        percentage: int, 
        sent: str, 
        ratio: float,
        min_ratio: float,
        max_ratio: float,
        score: float, 
        size: int=15
    ) -> str:

        if ratio == min_ratio:
            incremental_value = "0%"
        else:
            incremental_value = f"+{int(100*ratio/min_ratio-100)}%"

        html = f"""
        <div id="myturn">
            <span data-value="{percentage/2}" style="width:{percentage/2}%;">
                <strong style="color:white;padding-left: 0.7em;">{incremental_value}</strong>
            </span>
            <progress value="{percentage}" max="100"></progress>
            <p style='font-size:22px; padding:2px;'>{sent}</p>
        </div>
        """
        return html

    def __render(
        self, 
        sents: List[str],
        scores: List[float], 
        ratios: List[float]
    ) -> str:
        
        max_ratio = max(ratios)
        min_ratio = min(ratios)
        if max_ratio == min_ratio:
            ratio2percentage = lambda ratio: 1
        else:
            ratio2percentage = lambda ratio: int(12 + 88*((ratio - min_ratio) / (max_ratio - min_ratio)))

        html = ""
        for sent, ratio, score in zip(sents, ratios, scores):
            html += self.__progressbar(
                percentage=ratio2percentage(ratio), 
                sent=sent,
                ratio=ratio, 
                min_ratio=min_ratio,
                max_ratio=max_ratio,
                score=score
            )

        return self.html_head + html + self.html_footer
    
    def __getProportions(
        self, 
        scores: List[float],
    ) -> List[float]:
    
        min_score = min(scores)
        return [min_score/s for s in scores]

    def compute(
        self, 
        pll_dict: Dict[str, float],
    ) -> str:

        sorted_pll_dict = sorted(pll_dict.items(), key=lambda x: x[1], reverse=True)

        sents = [k for k,_ in sorted_pll_dict]
        scores  = [v for _,v in sorted_pll_dict]

        # Scape < and > marks from hightlight word/s
        translation_table = str.maketrans({"<": "<b><i>", ">": "</i></b>"})
        sents = [f"{i+1}. {s.translate(translation_table)}" for i, s in enumerate(sents)]
        ratios = self.__getProportions(scores)
        
        return self.__render(sents, scores, ratios)
