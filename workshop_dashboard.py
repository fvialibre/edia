#!/usr/bin/env python3
"""Live terminal dashboard for monitoring clinical patient consultations during workshops."""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

from data.clinical_patients.clinical_prompts import clinical_patients
from modules.module_patientAssignment import MEDICAL_PATIENT_POOL, PSYCHOLOGY_PATIENT_POOL

CLINICAL_ENCOUNTERS_PATH = "./logs/clinical_encounters.jsonl"
ACTIVE_ASSIGNMENTS_PATH = "./logs/active_assignments.json"

# ANSI color formatting
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"


def make_bar(count, max_val, color_code, width=14):
    if max_val <= 0 or count <= 0:
        return DIM + "░" * width + RESET
    filled = int(round((count / max_val) * width))
    filled = max(1, min(width, filled))
    return color_code + "█" * filled + DIM + "░" * (width - filled) + RESET


def render_dashboard():
    completed_counts = defaultdict(int)
    completed_by_token = defaultdict(set)
    recent_encounters = []
    total_encounters = 0

    # 1. Read completed encounters (ground truth on disk)
    if os.path.exists(CLINICAL_ENCOUNTERS_PATH):
        try:
            with open(CLINICAL_ENCOUNTERS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    total_encounters += 1
                    pid = entry.get("patient_id")
                    tok = str(entry.get("token_id", "")).strip().lower()
                    ts = entry.get("timestamp", "")
                    if pid:
                        completed_counts[pid] += 1
                        if tok:
                            completed_by_token[tok].add(pid)
                        recent_encounters.append({
                            "timestamp": ts,
                            "token_id": tok,
                            "patient_id": pid,
                        })
        except Exception:
            pass

    # 2. Read live active assignments (single source of truth from module_patientAssignment)
    active_by_token = {}
    if os.path.exists(ACTIVE_ASSIGNMENTS_PATH):
        try:
            with open(ACTIVE_ASSIGNMENTS_PATH, "r", encoding="utf-8") as f:
                raw_active = json.load(f)
                if isinstance(raw_active, dict):
                    for tok, pid in raw_active.items():
                        tok_clean = str(tok).strip().lower()
                        # Only active if not already recorded as completed on disk
                        if pid not in completed_by_token.get(tok_clean, set()):
                            active_by_token[tok_clean] = pid
        except Exception:
            pass

    # Calculate active counts per patient
    active_counts = defaultdict(int)
    for tok, pid in active_by_token.items():
        active_counts[pid] += 1

    # Max counts for scaling bars
    max_done_med = max([completed_counts[p] for p in MEDICAL_PATIENT_POOL] or [1])
    max_active_med = max([active_counts[p] for p in MEDICAL_PATIENT_POOL] or [1])
    max_done_psych = max([completed_counts[p] for p in PSYCHOLOGY_PATIENT_POOL] or [1])
    max_active_psych = max([active_counts[p] for p in PSYCHOLOGY_PATIENT_POOL] or [1])

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_active_students = len(active_by_token)
    total_finished_students = len(completed_by_token)

    lines = []
    lines.append(BOLD + CYAN + "╔════════════════════════════════════════════════════════════════════════════════════════════════╗" + RESET)
    lines.append(BOLD + CYAN + "║                       🏥 CLINICAL WORKSHOP LIVE MONITOR                                        ║" + RESET)
    lines.append(BOLD + CYAN + "╚════════════════════════════════════════════════════════════════════════════════════════════════╝" + RESET)
    lines.append(f"  {DIM}Last updated:{RESET} {BOLD}{now_str}{RESET}  │  {DIM}Active In-Flight:{RESET} {YELLOW}{BOLD}{total_active_students}{RESET} students  │  {DIM}Finished Consultations:{RESET} {GREEN}{BOLD}{total_encounters}{RESET}  │  {DIM}Unique Students:{RESET} {BOLD}{total_finished_students}{RESET}")
    lines.append("")

    def render_pool_table(title, title_color, pool, max_done, max_active):
        pool_done = sum(completed_counts[p] for p in pool)
        pool_active = sum(active_counts[p] for p in pool)
        lines.append(BOLD + title_color + f"── {title} ({len(pool)} cases) ── Total: {pool_done + pool_active} ({pool_done} done, {pool_active} active) " + "─" * 20 + RESET)
        lines.append(f"  {'ID':<18} {'Patient Name':<22} {'Done':>5}  {'Active':>6}  {'Total':>5}   {'Finished Bar (Done)':<18} {'Active Bar (In-Flight)':<18}")
        lines.append(DIM + "  " + "─" * 94 + RESET)
        for pid in pool:
            p_obj = clinical_patients.get(pid)
            name = p_obj.name if p_obj else pid
            d_count = completed_counts[pid]
            a_count = active_counts[pid]
            tot = d_count + a_count

            bar_done = make_bar(d_count, max(max_done, 1), GREEN, width=12)
            bar_active = make_bar(a_count, max(max_active, 1), YELLOW, width=12)

            lines.append(
                f"  {pid:<18} {name[:21]:<22} "
                f"{GREEN}{BOLD}{d_count:>5}{RESET}  "
                f"{YELLOW}{BOLD}{a_count:>6}{RESET}  "
                f"{BOLD}{tot:>5}{RESET}   "
                f"{bar_done} {GREEN}{d_count:>2}{RESET}   "
                f"{bar_active} {YELLOW}{a_count:>2}{RESET}"
            )
        lines.append("")

    # Medical Section
    render_pool_table("🩺 Medical Patient Pool", BLUE, MEDICAL_PATIENT_POOL, max_done_med, max_active_med)

    # Psychology Section
    render_pool_table("🧠 Psychology Patient Pool", MAGENTA, PSYCHOLOGY_PATIENT_POOL, max_done_psych, max_active_psych)

    # Active Participants Live Roster
    if active_by_token:
        lines.append(BOLD + YELLOW + f"── 👥 Currently Active Consultations ({len(active_by_token)} in progress) ──────────────────────────────" + RESET)
        active_items = []
        for tok, pid in sorted(active_by_token.items()):
            p_obj = clinical_patients.get(pid)
            p_name = p_obj.name if p_obj else pid
            active_items.append(f"  {CYAN}{tok:<16}{RESET} -> {BOLD}{p_name}{RESET} ({pid})")
        # Display up to 10 active
        lines.extend(active_items[:10])
        if len(active_items) > 10:
            lines.append(f"  {DIM}... and {len(active_items) - 10} more active students{RESET}")
        lines.append("")

    # Participant Breakdown
    rounds_dist = defaultdict(int)
    for tok, p_set in completed_by_token.items():
        rounds_dist[len(p_set)] += 1

    lines.append(BOLD + "── 📊 Completed Consultations per Student ──────────────────────────────────────────────" + RESET)
    dist_items = []
    for num_rounds in sorted(rounds_dist.keys()):
        dist_items.append(f"{num_rounds} case{'s' if num_rounds > 1 else ''}: {BOLD}{rounds_dist[num_rounds]}{RESET} students")
    lines.append("  " + ("  │  ".join(dist_items) if dist_items else "No completed consultations yet."))
    lines.append("")

    # Recent completions
    lines.append(BOLD + "── 🕒 Recent Completions ───────────────────────────────────────────────────────────────" + RESET)
    recent = recent_encounters[-5:]
    if recent:
        for r in reversed(recent):
            p_obj = clinical_patients.get(r["patient_id"])
            p_name = p_obj.name if p_obj else r["patient_id"]
            ts = r["timestamp"]
            try:
                dt = datetime.fromisoformat(ts)
                time_disp = dt.strftime("%H:%M:%S")
            except Exception:
                time_disp = ts[:19]
            lines.append(f"  {DIM}{time_disp}{RESET}  Token: {CYAN}{r['token_id']:<15}{RESET} -> {BOLD}{p_name}{RESET} ({r['patient_id']})")
    else:
        lines.append("  No completions recorded yet.")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Live clinical chatbot workshop monitor")
    parser.add_argument("-w", "--watch", action="store_true", help="Auto-refresh display every N seconds")
    parser.add_argument("-n", "--interval", type=int, default=2, help="Refresh interval in seconds (default: 2)")
    args = parser.parse_args()

    if not args.watch:
        print(render_dashboard())
        return

    try:
        while True:
            sys.stdout.write("\033[H\033[J")
            sys.stdout.write(render_dashboard())
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
