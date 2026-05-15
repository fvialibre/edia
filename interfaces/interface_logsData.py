import json
import gradio as gr
from matplotlib import pyplot as plt
from gradio_i18n import gettext as i18n


LOG_FILES_PATHS = {
    "TypicalPhrasesTab": "logs/logs_typicalPhrases.jsonl",
    "ChatbotTab": "logs/logs_chatbot.jsonl",
    "AmbiguousReferencesTab": "logs/logs_ambiguous_references.jsonl",
    "CVQATab": "logs/logs_cvqa.jsonl",
    "ValidatorTab": "logs/logs_validator.jsonl",
}


def _count_submissions(token_id: str) -> dict:
    counts = {}
    for tool, path in LOG_FILES_PATHS.items():
        count = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if str(record.get("token_id", "")) == token_id:
                        count += 1
        except FileNotFoundError:
            pass
        counts[tool] = count
    return counts


def interface(
    token_id, age, gender, nationality, region, school, consent_checkbox
) -> gr.Blocks:

    iface = gr.Blocks()

    def get_submission_counts(tid):
        tid = (tid or "").strip()
        if not tid:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, i18n("LogsDataNoTokenId"),
                    ha="center", va="center", fontsize=13)
            ax.axis("off")
            plt.tight_layout()
            plt.close()
            return fig

        raw_counts = _count_submissions(tid)
        # Translate tool keys for display
        counts = {i18n(k): v for k, v in raw_counts.items()}
        total = sum(counts.values())

        fig, ax = plt.subplots(figsize=(8, 4))
        tools = list(counts.keys())
        values = list(counts.values())
        colors = ["#4C72B0" if v > 0 else "#cccccc" for v in values]

        bars = ax.barh(tools, values, color=colors)
        ax.set_xlabel(i18n("LogsDataXLabel"))
        ax.set_title(i18n("LogsDataChartTitle").format(total=total))
        ax.invert_yaxis()

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.2,
                bar.get_y() + bar.get_height() / 2,
                str(val),
                va="center", ha="left", fontsize=11
            )

        ax.set_xlim(0, max(values) * 1.2 + 1)
        ax.tick_params(axis="y", labelsize=11)
        plt.tight_layout()
        plt.close()
        return fig

    with iface:
        gr.Markdown("## " + i18n("LogsDataTitle"))
        gr.Markdown(i18n("LogsDataDescription"))

        btn = gr.Button(value=i18n("LogsDataButton"))
        output = gr.Plot(show_label=False)

        btn.click(get_submission_counts, inputs=[token_id], outputs=[output])

    return iface