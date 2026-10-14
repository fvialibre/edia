import time
import threading
from gradio_client import Client
from tqdm import tqdm
from collections import deque
from datetime import datetime

client = Client("https://edia.ngrok.app/")

def make_request():
     try:
          result = client.predict(
               token_id="Hello!!",
               age=0,
               gender="M",
               nationality_personal_info=["Afghanistan"],
               personal_region_dropdown=[],
               consent_checkbox=False,
               data_point_phrase="DataPointPhrasePrefix `pincharrata`",
               data_point_definition="DataPointMeaningPrefix Explorador, boy scout (futb.) Club Atlético Estudiantes de la Plata.",
               new_phrase="Bae",
               new_phrase_definition="Before anyone else (acronym).",
               new_phrase_sentence_example="My bae and I are going to the movies tonight.",
               api_name="/on_llm_responses_button"
          )
          print(f"Success: {result}")
     except Exception as e:
          print(f"Error: {e}")

# Track metrics
completed_requests = 0
total_requests = 2000
request_times = deque(maxlen=10)  # Last 10 response times
start_time = time.time()
lock = threading.Lock()
pbar = tqdm(total=total_requests, desc="Requests", unit="req")

def make_request():
     global completed_requests
     req_start = time.time()
     try:
          result = client.predict(
               token_id="Hello!!",
               age=0,
               gender="M",
               nationality_personal_info=["Afghanistan"],
               personal_region_dropdown=[],
               consent_checkbox=False,
               data_point_phrase="DataPointPhrasePrefix `pincharrata`",
               data_point_definition="DataPointMeaningPrefix Explorador, boy scout (futb.) Club Atlético Estudiantes de la Plata.",
               new_phrase="Bae",
               new_phrase_definition="Before anyone else (acronym).",
               new_phrase_sentence_example="My bae and I are going to the movies tonight.",
               api_name="/on_llm_responses_button"
          )
          response_time = time.time() - req_start
          with lock:
               completed_requests += 1
               request_times.append(response_time)
               avg_delay = sum(request_times) / len(request_times)
               remaining = total_requests - completed_requests
               elapsed = time.time() - start_time
               pbar.update(1)
               pbar.set_postfix({
                    "avg_delay": f"{avg_delay:.2f}s",
                    "remaining": remaining,
                    "elapsed": f"{elapsed:.1f}s"
               })
     except Exception as e:
          with lock:
               pbar.write(f"Error: {e}")
requests_per_minute = 300
interval = 60 / requests_per_minute

for i in range(2000):  # Adjust range for duration needed
     threading.Thread(target=make_request, daemon=True).start()
     time.sleep(interval)

time.sleep(10)  # Wait for responses