import gradio as gr
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from auth import school_list
from wordcloud import WordCloud
import os
import ast
from nltk.corpus import stopwords
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
    )["LogsData_interface"]

    
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

    def get_plots(selected_school, selected_graphs):
        if (selected_school is None 
            or selected_graphs is None
            or selected_school is None
            or selected_school not in school_list
            or len(selected_graphs) == 0
        ):
            plt.text(0.5, 0.5, 'Tenés que seleccionar tu escuela y por lo menos un gráfico', ha='center', va='center', fontsize=12)
            plt.axis('off')
            fig = plt.gcf()
            plt.close()
            return fig
        
        # --- Load CSV into pandas DataFrame ---
        df = pd.read_csv("logs/logs_edia_lmodels_biasphrase_es.csv")
        selected_school_df = df[df['school'] == selected_school]
        if selected_school_df.shape[0] == 0:
            plt.text(0.5, 0.5, 'No hay registros para esta escuela', ha='center', va='center', fontsize=12)
            plt.axis('off')
            fig = plt.gcf()
            plt.close()
            return fig

        bias_type_df = selected_school_df.copy()

        if ("Tipo de sesgo" in selected_graphs
            or "Tipo de sesgo por género" in selected_graphs):
            
            bias_type_df['lista_de_palabras'] = bias_type_df['lista_de_palabras'].str.replace('[\'', '').str.replace('\']', '').str.split("\', \'")
            bias_type_df['tipo_de_sesgo_explorado'] = bias_type_df['tipo_de_sesgo_explorado'].str.split(",")
            bias_type_df = bias_type_df.explode('lista_de_palabras', ignore_index=True)
            bias_type_df = bias_type_df.explode('tipo_de_sesgo_explorado', ignore_index=True)
            bias_type_df['tipo_de_sesgo_explorado'] = bias_type_df['tipo_de_sesgo_explorado'].str.replace('[', '').str.replace(']', '').str.replace('\'', '').str.strip()

        # make subplots for each selected graph
        fig, axs = plt.subplots(len(selected_graphs), 1, squeeze=False, figsize=(12, len(selected_graphs) * 6))
        axs = axs.flatten()
        fig.subplots_adjust(hspace=0.5, wspace=0.5)

        for i, graph in enumerate(selected_graphs):
            if graph == "Cantidad de frases por escuela":
                school_counts = df['school'].value_counts().reset_index()
                school_counts.columns = ['school', 'count']

                top_n = school_counts.head(15)

                if selected_school not in top_n['school'].values:
                    if selected_school not in school_counts['school'].values:
                        top_n = pd.concat([top_n, pd.DataFrame({'school': [selected_school], 'count': [0]})])
                    else:
                        top_n = pd.concat([top_n, school_counts[school_counts['school'] == selected_school]])

                colors = ['blue' if school == selected_school else 'gray' for school in top_n['school']]
                top_n.loc[:, 'school'] = [f"Escuela {chr(65+i)}" if school != selected_school else divide_string(school_list[school]) for i, school in enumerate(top_n['school'])]

                axs[i].barh(top_n['school'], top_n['count'], color=colors)
                axs[i].set_xlabel('Número de registros')
                axs[i].set_ylabel('Escuela')
                axs[i].set_title('Escuelas con más registros')
                axs[i].tick_params(axis='x', labelsize=8)
                axs[i].tick_params(axis='y', labelsize=8)
                axs[i].invert_yaxis()             
            elif graph == "Tipo de sesgo":
                tipo_de_sesgo_counts = bias_type_df['tipo_de_sesgo_explorado'].value_counts()
                tipo_de_sesgo_percentages = (tipo_de_sesgo_counts / tipo_de_sesgo_counts.sum()) * 100

                # Select the top 9 options for each gender and group the rest as "Otros"
                threshold = 5
                top_tipo_de_sesgo_percentages = tipo_de_sesgo_percentages[tipo_de_sesgo_percentages >= threshold]

                # Calculate the percentage for "Otros" category
                other_tipo_de_sesgo_percentage = tipo_de_sesgo_percentages[tipo_de_sesgo_percentages < threshold].sum()

                # Add "Otros" percentage to the top percentages
                top_tipo_de_sesgo_percentages['Otros'] = other_tipo_de_sesgo_percentage
                # Plot the pie chart in the subplot
                axs[i].pie(top_tipo_de_sesgo_percentages, labels=top_tipo_de_sesgo_percentages.index, autopct='%1.1f%%', startangle=90, counterclock=False)
                axs[i].axis('off')
                axs[i].set_title('Porcentaje de sesgos explorados')
            elif graph == "Tipo de sesgo por género":
                # Filter the dataframe for each gender
                df_m = bias_type_df[bias_type_df['gender'] == 'M']
                df_f = bias_type_df[bias_type_df['gender'] == 'F']
                df_x = bias_type_df[bias_type_df['gender'] == 'X']

                # Calculate the value counts for each gender
                m_counts = df_m['tipo_de_sesgo_explorado'].value_counts()
                f_counts = df_f['tipo_de_sesgo_explorado'].value_counts()
                x_counts = df_x['tipo_de_sesgo_explorado'].value_counts()

                # Calculate the value counts for each gender as percentage
                m_percentages = (m_counts / m_counts.sum()) * 100 if not m_counts.empty else pd.Series()
                f_percentages = (f_counts / f_counts.sum()) * 100 if not f_counts.empty else pd.Series()
                x_percentages = (x_counts / x_counts.sum()) * 100 if not x_counts.empty else pd.Series()

                # Select the top 9 options for each gender and group the rest as "Otros"
                threshold = 5
                top_m_percentages = m_percentages[m_percentages >= threshold]
                top_f_percentages = f_percentages[f_percentages >= threshold]
                top_x_percentages = x_percentages[x_percentages >= threshold]

                # Calculate the percentage for "Otros" category
                other_m_percentage = m_percentages[m_percentages < threshold].sum() if not m_percentages.empty else 0
                other_f_percentage = f_percentages[f_percentages < threshold].sum() if not f_percentages.empty else 0
                other_x_percentage = x_percentages[x_percentages < threshold].sum() if not x_percentages.empty else 0

                # Add "Otros" percentage to the top percentages
                if not m_percentages.empty:
                    top_m_percentages['Otros'] = other_m_percentage
                if not f_percentages.empty:
                    top_f_percentages['Otros'] = other_f_percentage
                if not x_percentages.empty:
                    top_x_percentages['Otros'] = other_x_percentage

                # Create subplots for the pie charts within the global subplot
                inner_fig, inner_axs = plt.subplots(1, 3, figsize=(15, 5))

                # Plot the pie chart for male gender
                if m_percentages.empty:
                    inner_axs[0].text(0.5, 0.5, 'No hay datos de género M', ha='center', va='center', fontsize=12)
                    inner_axs[0].axis('off')   
                elif df_m['token_id'].nunique() <= 1:
                    inner_axs[0].text(0.5, 0.5, 'No hay suficientes datos de género M', ha='center', va='center', fontsize=12)
                    inner_axs[0].axis('off')
                else:
                    inner_axs[0].pie(top_m_percentages, labels=top_m_percentages.index, autopct='%1.1f%%', startangle=90, counterclock=False)
                    inner_axs[0].set_title(f'Género M - {df_m.shape[0]} frases')

                # Plot the pie chart for female gender
                if f_percentages.empty:
                    inner_axs[1].text(0.5, 0.5, 'No hay datos de género F', ha='center', va='center', fontsize=12)
                    inner_axs[1].axis('off')   
                elif df_f['token_id'].nunique() <= 1:
                    inner_axs[1].text(0.5, 0.5, 'No hay suficientes datos de género F', ha='center', va='center', fontsize=12)
                    inner_axs[1].axis('off')
                else:
                    inner_axs[1].pie(top_f_percentages, labels=top_f_percentages.index, autopct='%1.1f%%', startangle=90, counterclock=False)
                    inner_axs[1].set_title(f'Género F - {df_f.shape[0]} frases')

                # Plot the pie chart for other gender
                if x_percentages.empty:
                    inner_axs[2].text(0.5, 0.5, 'No hay datos de género X', ha='center', va='center', fontsize=12)
                    inner_axs[2].axis('off')   
                elif df_x['token_id'].nunique() <= 1:
                    inner_axs[2].text(0.5, 0.5, 'No hay suficientes datos de género X', ha='center', va='center', fontsize=12)
                    inner_axs[2].axis('off')
                else:
                    inner_axs[2].pie(top_x_percentages, labels=top_x_percentages.index, autopct='%1.1f%%', startangle=90, counterclock=False)
                    inner_axs[2].set_title(f'Género X - {df_x.shape[0]} frases')

                # Add the inner figure to the global subplot
                # Convert the inner figure to an image array
                inner_fig.canvas.draw()
                inner_img = np.frombuffer(inner_fig.canvas.tostring_rgb(), dtype=np.uint8)
                inner_img = inner_img.reshape(inner_fig.canvas.get_width_height()[::-1] + (3,))

                # Display the inner image in the global subplot
                axs[i].imshow(inner_img)
                axs[i].axis('off')
                axs[i].set_title('Distribución de tipos de sesgo por género')

                # Close the inner figure to free up memory
                plt.close(inner_fig)
            elif graph == "Combinaciones de tipo de sesgo y género más comunes":
                # Correct the apply to work on grouped_df
                bias_type_gender_df = bias_type_df.groupby(['tipo_de_sesgo_explorado', 'gender']).size().reset_index(name='count')
                bias_type_gender_df['tipo_de_sesgo_explorado-gender'] = bias_type_gender_df.apply(lambda row: row['tipo_de_sesgo_explorado'] + ' - ' + row['gender'], axis=1)

                # Sorting and getting top 10, but based on count now
                bias_type_gender_df = bias_type_gender_df.sort_values(by='count', ascending=False).head(10)
                bias_type_gender_df = bias_type_gender_df[::-1]

                # Plotting in the ith subplot
                axs[i].barh(bias_type_gender_df['tipo_de_sesgo_explorado-gender'], bias_type_gender_df['count'])
                axs[i].set_xlabel('Número de frases registradas')
                axs[i].set_ylabel('Combinación de Tipo de sesgo y Género')
                axs[i].set_title('Combinaciones de tipo de sesgo y género más comunes')
            elif graph == "Nube de palabras":
                # Prepare the text
                frases = " ".join(frase for frase in selected_school_df.frase)

                stop_words = set(stopwords.words('spanish'))  # or any other language
                frases = " ".join([word for word in frases.split() if word.lower() not in stop_words])

                # Create the word cloud
                word_cloud1 = WordCloud(
                    collocations=False,
                    background_color='white',
                    width=2048,
                    height=1080,
                ).generate(frases)

                # Display the word cloud as a subplot
                axs[i].imshow(word_cloud1, interpolation='bilinear')
                axs[i].axis('off')
                axs[i].set_title('Nube de palabras')
            elif graph == "Frases con mayor diferencia de métrica":
                selected_school_df = selected_school_df.copy()
                selected_school_df['results'] = selected_school_df['results'].apply(ast.literal_eval)

                def get_max_relative_diff(results):
                    max_value = max(results.values())
                    min_value = min(results.values())
                    relative_diff = max_value / min_value

                    if relative_diff < 1:
                        relative_diff = 1 / relative_diff

                    return relative_diff

                selected_school_df['max_relative_diff'] = selected_school_df['results'].apply(get_max_relative_diff)

                # sort the dataframe by max_diff in descending order
                selected_school_df = selected_school_df.sort_values(by='max_relative_diff', ascending=False)
                top_n_relative_diff = selected_school_df.head(5)

                # print the max and min keys and values for each row
                for _, row in top_n_relative_diff.iterrows():
                    results = row['results']
                    max_key = max(results, key=results.get)
                    min_key = min(results, key=results.get)
                    relative_diff = round(row['max_relative_diff'], 2)
                    bar = axs[i].barh(f"Max: {max_key}\nMin: {min_key}", relative_diff, color='blue')
                    # Write the y tick inside the bar, left aligned, starting at x=0
                    axs[i].text(0.05, bar[0].get_y() + bar[0].get_height() / 2, 
                                f"Max: {max_key}\nMin: {min_key}", ha='left', va='center', color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.7))
                axs[i].set_xlabel('Diferencia relativa')
                axs[i].set_title('Las 5 frases con mayor diferencia relativa')
                axs[i].invert_yaxis()
                axs[i].tick_params(axis='y', which='both', length=0, labelsize=0)
                axs[i].set_yticks([])
            elif graph == "Tamaño de vocabulario":
                words = set()
                for _, row in selected_school_df.iterrows():
                    words.update(row['frase'].split())
                    words.update(row['lista_de_palabras'])
                axs[i].text(0.5, 0.5, f'Se usaron {len(words)} palabras distintas.', ha='center', va='center', fontsize=12)
                axs[i].axis('on')
                axs[i].tick_params(axis='both', which='both', length=0, labelsize=0)
                axs[i].set_title('Tamaño de vocabulario')
            elif graph == "Tiempo activo en la plataforma":
                # get how many 15 minute time slots have entries
                selected_school_df = selected_school_df.copy()
                selected_school_df['formatted_time'] = pd.to_datetime(selected_school_df['datetime'], dayfirst=True)

                # Round the time to the nearest round_to minutes
                round_to = 5

                selected_school_df.loc[:, 'formatted_time'] = selected_school_df['formatted_time'].dt.floor(f'{round_to}min')
                total_minutes = round_to * selected_school_df['formatted_time'].nunique()
                total_hours = total_minutes / 60
                
                axs[i].text(0.5, 0.5, f"EDIA fue usado por {total_minutes} minutos ({total_hours:.2f} horas), contando intervalos activos de {round_to} minutos.", 
                            ha='center', va='center', fontsize=12)
                axs[i].axis('on')
                axs[i].tick_params(axis='both', which='both', length=0, labelsize=0)
                axs[i].set_title('Tiempo activo en la plataforma')
            
        plt.tight_layout()
        fig = plt.gcf()
        plt.close()
        return fig

    with iface:
        with gr.Row():
            with gr.Column():
                school = gr.Number(
                    value=0,
                    label=labels["school"],
                )
                school_name = gr.HTML(
                    value=labels["notschool"],
                )
            with gr.Column():
                _ = gr.HTML(
                    value=labels["ref"],
                )
        # make checkbox buttons to select graphs
        graph_selection = gr.CheckboxGroup(
            labels["boxgroup"],
            label=labels["graphsel"],
        )
        btn = gr.Button(value=labels["button1"])

        output = gr.Plot(show_label=False)
        download_btn = gr.Button(value=labels["button2"], visible=False)

        def filter_and_download_csv(selected_school):
            df = pd.read_csv("logs/logs_edia_lmodels_biasphrase_es.csv")
            filtered_df = df[df['school'] == selected_school]
            filtered_df = filtered_df.drop(columns=['token_id'])  # Drop the token_id column
            # Create directory if it doesn't exist
            save_dir = "logs/downloaded_CSVs"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_path = f"{save_dir}/{selected_school}_filtered.csv"
            filtered_df.to_csv(save_path, index=False)
            return save_path

        def update_school_name(school):
            if school is None or school == 0:
                return (
                    gr.HTML(
                        value=labels["notschool"],
                    )
                )
            elif school not in school_list:
                return (
                    gr.HTML(
                        value=labels["notexistschool"],
                    )
                )
            else:
                return (
                    gr.HTML(
                        value=labels["schoolsel"],
                    )
                )
        school.change(
            fn=update_school_name,
            inputs=[
                school
            ],
            outputs=[school_name])
        
        def update_download_btn_visibility(selected_school):
            return gr.update(visible=bool(selected_school))

        school.change(update_download_btn_visibility, [school], [download_btn])
        download_btn.click(filter_and_download_csv, [school], gr.File())
        btn.click(get_plots, [school, graph_selection], output)
    return iface