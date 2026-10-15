# Image URIs
NAVBAR_LOGO_URI = "https://i.imgur.com/b0EkuNi.jpeg"
FOOTER_LOGO_URI = "https://i.imgur.com/CiQNtey.jpeg"
CLINICAL_CHATBOT_BACKGROUND_URI = "https://i.imgur.com/NQfqr2f.jpeg"
CLINICAL_CHATBOT_PATIENT_WAITING_URI = "https://i.imgur.com/JDSFVDs.jpeg"

# Head HTML tags to prevent unwanted browser auto-translation
HEAD_TAGS = """
<meta name="google" content="notranslate">
<meta name="robots" content="notranslate">
<script>
    if (typeof document !== 'undefined' && document.documentElement) {
        document.documentElement.setAttribute('translate', 'no');
        document.documentElement.classList.add('notranslate');
    }
</script>
"""

# Layout HTML components
NAVBAR_HTML = f"""
    <div translate="no" class="notranslate" style="background-color: black; color: white; padding: 10px; margin-bottom: 30px; width: 100%; display: flex; justify-content: center; align-items: center;">
        <a href="https://edia.ngrok.app/" style="display: flex; justify-content: center; align-items: center;">
            <img src="{NAVBAR_LOGO_URI}" style="max-width: 20vw; width: 11em; height: auto; display: block;">
        </a>
    </div>
    <script>
        (function() {{
            function disableTranslation() {{
                if (typeof document !== 'undefined' && document.documentElement) {{
                    document.documentElement.setAttribute('translate', 'no');
                    document.documentElement.classList.add('notranslate');
                }}
                if (typeof document !== 'undefined') {{
                    document.querySelectorAll('.notranslate').forEach(function(el) {{
                        el.setAttribute('translate', 'no');
                    }});
                }}
            }}
            disableTranslation();
            if (typeof MutationObserver !== 'undefined' && typeof document !== 'undefined') {{
                var observer = new MutationObserver(disableTranslation);
                var target = document.body || document.documentElement;
                if (target) {{
                    observer.observe(target, {{ childList: true, subtree: true }});
                }}
            }}
        }})();
    </script>
"""

FOOTER_HTML = f"""
    <div style="width: 100%; margin-top: 30px; margin-bottom: 20px; padding: 10px 0; display: flex; justify-content: center; align-items: center; clear: both;">
        <img src="{FOOTER_LOGO_URI}">
    </div>
"""

HTML_FEEDBACK_TITLE = """
    <h1 style="text-align: center; margin-bottom: 0em;">
        Feedback del mensaje<br>
    </h1>
    <h3 style="text-align: center; margin-bottom: 3em;">
        Por favor responda las siguientes preguntas sobre el mensaje que seleccionó.
    </h3>
"""

HTML_HR = "<hr>"

def build_header_title_html(title):
    return "<h1 style='text-align: center;'>" + title + "</h1>"

def build_end_conversation_modal_header_html(title, description):
    return (
        "<h1 style='text-align: center; margin-bottom: 0em;'>"
        + title
        + "</h1><h3 style='text-align: center; margin-bottom: 1em;'>"
        + description
        + "</h3>"
    )

def build_patient_card_empty_html(patient_label, no_patient_text):
    return (
        '<div style="display: flex; align-items: center; gap: 14px; padding: 12px 18px; '
        'border-radius: 10px; border: 1.5px dashed rgba(156, 163, 175, 0.5); '
        'background: rgba(156, 163, 175, 0.07); min-height: 52px; box-sizing: border-box;">'
        '<span style="font-size: 1.7em; line-height: 1;">🚪</span>'
        '<div>'
        f'<div style="font-size: 0.75em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.7;">{patient_label}</div>'
        f'<div style="font-size: 1.05em; font-weight: 500; opacity: 0.65; font-style: italic;">{no_patient_text}</div>'
        '</div>'
        '</div>'
    )

def build_patient_card_active_html(in_consultation_text, patient_name, in_consultation_status):
    return (
        '<div style="display: flex; align-items: center; justify-content: space-between; gap: 14px; padding: 12px 18px; '
        'border-radius: 10px; border: 1.5px solid rgba(59, 130, 246, 0.55); '
        'background: rgba(59, 130, 246, 0.08); min-height: 52px; box-sizing: border-box;">'
        '<div style="display: flex; align-items: center; gap: 14px;">'
        '<span style="font-size: 1.7em; line-height: 1;">🧑‍⚕️</span>'
        '<div>'
        f'<div style="font-size: 0.75em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #3b82f6;">{in_consultation_text}</div>'
        f'<div style="font-size: 1.3em; font-weight: 700; letter-spacing: -0.01em;">{patient_name}</div>'
        '</div>'
        '</div>'
        '<div style="display: flex; align-items: center; gap: 6px; background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.35); color: #15803d; padding: 6px 12px; border-radius: 20px; font-size: 0.82em; font-weight: 600;">'
        '<span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: #22c55e; box-shadow: 0 0 6px #22c55e;"></span>'
        f'<span>{in_consultation_status}</span>'
        '</div>'
        '</div>'
    )

def build_chat_placeholder_html(image_uri, text):
    return (
        '<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; width: 100%; margin: 0 auto; padding: 1.5em 1em;">'
        f'<img src="{image_uri}" '
        'style="max-width: min(100%, 480px); height: auto; border-radius: 12px; margin: 0 auto 1.25em auto; display: block; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);" />'
        f'<div style="text-align: center; max-width: 480px; margin: 0 auto;">{text}</div>'
        '</div>'
    )

# CSS Styles
CLINICAL_CHATBOT_CSS = f"""
.contain {{ display: flex !important; flex-direction: column !important; }}
#component-0, #component-3, #component-10, #component-8 {{ height: 100% !important; }}
#chatbot {{ flex-grow: 1 !important; overflow: auto !important; }}
#chatbot .placeholder-content {{ display: flex !important; justify-content: center !important; align-items: center !important; width: 100% !important; height: 100% !important; text-align: center !important; }}
#chatbot .placeholder {{ display: flex !important; justify-content: center !important; align-items: center !important; width: 100% !important; height: 100% !important; text-align: center !important; margin: auto !important; }}
#chatbot .placeholder * {{ text-align: center !important; margin-left: auto !important; margin-right: auto !important; }}
@media (min-width: 769px) {{
  #col {{ height: calc(100vh - 240px) !important; min-height: 520px !important; }}
  #complementary-studies-col {{
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 12px;
    overflow-y: auto !important;
    max-height: 100% !important;
    box-sizing: border-box !important;
  }}
}}
@media (max-width: 768px) {{
  #col {{ height: auto !important; min-height: auto !important; }}
  #col > div {{ height: auto !important; }}
  #chatbot {{ height: 380px !important; max-height: 50vh !important; }}
  #complementary-studies-col {{
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 12px;
    height: auto !important;
    max-height: none !important;
    overflow: visible !important;
    margin-top: 14px !important;
    margin-bottom: 14px !important;
    box-sizing: border-box !important;
  }}
}}
#end-conversation-button {{
  width: 100% !important;
  margin-top: 14px !important;
  margin-bottom: 6px !important;
}}
@keyframes clinical-bounce-arrow {{
  0%, 100% {{ transform: translateY(0); }}
  50% {{ transform: translateY(4px); }}
}}
.clinical-bounce-arrow {{
  animation: clinical-bounce-arrow 1.3s infinite ease-in-out;
}}
@keyframes pulse-chat-input {{
  0% {{
    box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.55);
    border-color: rgba(59, 130, 246, 0.7);
  }}
  50% {{
    box-shadow: 0 0 0 6px rgba(59, 130, 246, 0.18);
    border-color: #3b82f6;
  }}
  100% {{
    box-shadow: 0 0 0 0 rgba(59, 130, 246, 0);
    border-color: rgba(59, 130, 246, 0.7);
  }}
}}
#clinical-chat-textbox textarea:not(:disabled), #clinical-chat-textbox input:not(:disabled) {{
  animation: pulse-chat-input 2.4s infinite ease-in-out;
  border: 1.5px solid #3b82f6 !important;
}}
#clinical-chat-textbox textarea:not(:disabled):focus, #clinical-chat-textbox input:not(:disabled):focus {{
  animation: none !important;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.35) !important;
}}
#physical-exam-text,
#complementary-studies-text {{
  max-height: min(240px, 32vh) !important;
  overflow-y: auto !important;
  border: 1px solid var(--border-color-primary, rgba(156, 163, 175, 0.3)) !important;
  border-radius: 8px !important;
  padding: 8px 12px !important;
  background: var(--background-fill-secondary, rgba(156, 163, 175, 0.05)) !important;
  margin-top: 6px !important;
  margin-bottom: 8px !important;
  box-sizing: border-box !important;
}}
#physical-exam-text.hide,
#complementary-studies-text.hide {{
  display: none !important;
}}
#physical-exam-text .prose,
#complementary-studies-text .prose {{
  max-width: 100% !important;
  max-height: none !important;
  overflow: visible !important;
  border: none !important;
  background: transparent !important;
  padding: 0 !important;
  margin: 0 !important;
}}
#physical-exam-text table,
#complementary-studies-text table {{
  width: 100% !important;
  margin-bottom: 6px !important;
  overflow: visible !important;
}}
#physical-exam-text::-webkit-scrollbar,
#complementary-studies-text::-webkit-scrollbar {{
  width: 6px;
}}
#physical-exam-text::-webkit-scrollbar-thumb,
#complementary-studies-text::-webkit-scrollbar-thumb {{
  background: rgba(156, 163, 175, 0.5);
  border-radius: 4px;
}}
"""

css = f"""
#small span{{
font-size: 8em;
}}
#examples {{color: black !important}}
#app-title {{ padding-left: 0.8em; }}
#app-personal-data-missing-heading {{ padding-left: 0.8em; }}
#end-conversation-button, #end-conversation-button button,
#sample-patient-button, #sample-patient-button button {{
    --button-cancel-background-fill: #dc2626 !important;
    --button-cancel-background-fill-hover: #b91c1c !important;
    --button-cancel-border-color: #dc2626 !important;
    --button-cancel-border-color-hover: #b91c1c !important;
    --button-cancel-text-color: #ffffff !important;
    background: #dc2626 !important;
    border-color: #dc2626 !important;
    color: white !important;
}}
#end-conversation-button:hover, #end-conversation-button button:hover,
#sample-patient-button:hover, #sample-patient-button button:hover {{
    background: #b91c1c !important;
    border-color: #b91c1c !important;
}}
{CLINICAL_CHATBOT_CSS}
"""
