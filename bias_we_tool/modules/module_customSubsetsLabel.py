class CustomSubsetsLabel:
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
                    progress {
                        width:100%;
                        height:4px;
                        border-radius: 1px;
                    }
                    #myturn {
                        display: block;
                        position: relative;
                        margin: auto;
                        width: 90%;
                        padding: 2px;
                    }
                </style>
            </head>
            <body>
        """

        self.html_footer ="</body></html>"

        self.subset_links = {
            'allwikis': "https://github.com/josecannete/wikiextractorforBERT",
            'DGT': "http://opus.nlpl.eu/DGT.php",
            'DOGC': "http://opus.nlpl.eu/DOGC.php",
            'ECB': "http://opus.nlpl.eu/ECB.php",
            'EMEA': "http://opus.nlpl.eu/EMEA.php",
            'EUBookShop': "http://opus.nlpl.eu/EUbookshop.php",
            'Europarl': "http://opus.nlpl.eu/Europarl.php",
            'GlobalVoices': "http://opus.nlpl.eu/GlobalVoices.php",
            'JRC': "http://opus.nlpl.eu/JRC-Acquis.php",
            'multiUN': "http://opus.nlpl.eu/MultiUN.php",
            'NewsCommentary11': "http://opus.nlpl.eu/News-Commentary-v11.php",
            'OpenSubtitles2018': "http://opus.nlpl.eu/OpenSubtitles-v2018.php",
            'ParaCrawl': "http://opus.nlpl.eu/ParaCrawl.php",
            'TED': "http://opus.nlpl.eu/TED2013.php",
            'UN': "http://opus.nlpl.eu/UN.php",
        }

    def __progressbar(self, percentage, subset, freq, size=15):
        html = f"""
        <div id="myturn">
            <progress value="{int(percentage)}" max="100"></progress>
            <p style="text-align:left; font-size:{size}px; padding:0px;">
                <a href="{self.subset_links[subset]}" target="_blank">
                    <strong>{subset}</strong> <span style="font-size:{size-2}px">(Frecuencia: {freq})</span>
                </a>
                <span style="float:right;">
                    <strong>{percentage}%</strong>
                </span>
            </p>
        </div>
        """
        return html

    def __render(self, subsets, freqs, percentages):
        html = ""
        for subset, freq, perc in zip(subsets, freqs, percentages):
            html += self.__progressbar(
                percentage=perc,
                subset=subset,
                freq=freq
            )

        return self.html_head + html + self.html_footer
    
    def compute(self, subsets_dic):
        subsets_dic_info = {
            k.split()[0]:{'freq':int(k.split()[1][1:-1]),'perc':round(v*100,2)} 
            for k,v in subsets_dic.items()
        }
        
        subsets = list(subsets_dic_info.keys())
        freqs = [d['freq'] for d in subsets_dic_info.values()]
        percentages = [d['perc'] for d in subsets_dic_info.values()]
        return self.__render(subsets, freqs, percentages)