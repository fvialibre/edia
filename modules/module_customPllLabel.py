class CustomPllLabel:
    def __init__(self):
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
    
    def __progressbar(self, percentage, sent, ratio, score, size=15):
        html = f"""
        <div id="myturn">
            <span data-value="{percentage/2}" style="width:{percentage/2}%;">
                <strong>x{round(ratio,3)}</strong> (score:{round(score,2)})
            </span>
            <progress value="{percentage}" max="100"></progress>
            <p style='font-size:22px; padding:2px;'>{sent}</p>
        </div>
        """
        return html

    def __render(self, sents, scores, ratios):
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
    
    def __getProportions(self, scores):
        min_score = min(scores)
        return [min_score/s for s in scores]

    def compute(self, pll_dict):
        sorted_pll_dict = dict(sorted(pll_dict.items(), key=lambda x: x[1], reverse=True))
        sents = list(sorted_pll_dict.keys())
        scores  = list(sorted_pll_dict.values())
        ratios = self.__getProportions(scores)

        return self.__render(sents, scores, ratios)