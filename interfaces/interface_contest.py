import gradio as gr
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from auth import school_list


def interface(
    available_logs: bool, 
    lang: str="es",
) -> gr.Blocks:

    # -- Load examples --
    if lang == 'es':
        from examples.examples_es import examples_sesgos_frases
    elif lang == 'en':
        from examples.examples_en import examples_sesgos_frases

    # --- Get language labels---
    labels = pd.read_json(
        f"language/{lang}.json"
    )["PhraseExplorer_interface"]

    
    # --- Init Interface ---
    iface = gr.Blocks(
        # css=".container {max-width: 90%; margin: auto;}"
    )

    def divide_string(s):
        # Split the string into words based on whitespace
        words = s.split()
        
        # Calculate the number of words in each part
        total_words = len(words)
        if total_words == 0:
            return ["", "", ""]
        
        # Determine the size of each part
        third = total_words // 3
        remainder = total_words % 3
        
        # Determine the end indices for each part
        idx1 = third + (1 if remainder > 0 else 0)
        idx2 = idx1 + third + (1 if remainder > 1 else 0)
        
        # Create the three parts
        part1 = ' '.join(words[:idx1])
        part2 = ' '.join(words[idx1:idx2])
        part3 = ' '.join(words[idx2:])
        
        # Return the parts joined by newlines
        return f"{part1}\n{part2}\n{part3}"

    def plot_ranking(selected_school):
        if len(selected_school) == 0:
            plt.text(0.5, 0.5, 'Tenés que seleccionar tu escuela', ha='center', va='center', fontsize=12)
            plt.axis('off')
            fig = plt.gcf()
            plt.close()
            return fig

        # --- Load CSV into pandas DataFrame ---
        df = pd.read_csv("/home/givetta/edia/logs/logs_edia_lmodels_biasphrase_es.csv")
        school_counts = df['school'].value_counts().reset_index()
        school_counts.columns = ['school', 'count']

        top_n = school_counts.head(15)

        if selected_school not in top_n['school'].values:
            if selected_school not in school_counts['school'].values:
                top_n = pd.concat([top_n, pd.DataFrame({'school': [selected_school], 'count': [0]})])
            else:
                top_n = pd.concat([top_n, school_counts[school_counts['school'] == selected_school]])

        colors = ['blue' if school == selected_school else 'gray' for school in top_n['school']]
        top_n.loc[:, 'school'] = [f"Escuela {chr(65+i)}" if school != selected_school else divide_string(school) for i, school in enumerate(top_n['school'])]

        # divide string in 3 parts with newlines. Only divide in whitespace


        top_n = top_n[::-1]
        colors = colors[::-1]

        plt.figure(figsize=(12, 8))
        plt.barh(top_n['school'], top_n['count'], color=colors)
        plt.xlabel('Número de registros')
        plt.ylabel('Escuela')
        plt.title('Escuelas con más registros')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=8, wrap=True)
        plt.tight_layout()
        fig = plt.gcf()
        plt.close()
        return fig
    
    with iface:
        school_name = gr.Dropdown(school_list, label="Escuela")
        btn = gr.Button(value="Buscar")

        output = gr.Plot(show_label=False)

        btn.click(plot_ranking, [school_name], output)
    
    return iface