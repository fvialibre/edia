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
                    progress::-webkit-progress-bar {
                        background-color: #666;
                        border-radius: 7px;
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
        score: float, 
        size: int=15
    ) -> str:

        html = f"""
        <div id="myturn">
            <span data-value="{percentage/2}" style="width:{percentage/2}%;">
                <strong>x{round(ratio,3)}</strong>
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
        ratio2percentage = lambda ratio: int(ratio*100/max_ratio)

        html = ""
        for sent, ratio, score in zip(sents, ratios, scores):
            html += self.__progressbar(
                percentage=ratio2percentage(ratio), 
                sent=sent,
                ratio=ratio, 
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
        pll_dict: Dict[str, float]
    ) -> str:

        sorted_pll_dict = dict(sorted(pll_dict.items(), key=lambda x: x[1], reverse=True))
        
        sents = list(sorted_pll_dict.keys())
        # Scape < and > marks from hightlight word/s
        sents = [s.replace("<","&#60;").replace(">","&#62;")for s in sents]

        scores  = list(sorted_pll_dict.values())
        ratios = self.__getProportions(scores)

        return self.__render(sents, scores, ratios)