# -*- coding: utf-8 -*-
import sys

def fix_manuscript():
    path = 'v3/manuscript_temporal/manuscript.tex'
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Define the exact text to replace
    # We will search from \subsection{Clinical Cohorts and Ethical Compliance} to \subsection{EEG Acquisition
    prefix = '\\subsection{Clinical Cohorts and Ethical Compliance}\n'
    suffix = '\n\\subsection{EEG Acquisition, Preprocessing, and Physical Adjacency Constraint}'
    
    start_idx = text.find(prefix)
    end_idx = text.find(suffix)
    assert start_idx != -1, "Prefix not found"
    assert end_idx != -1, "Suffix not found"

    new_section = r"""\subsection{Clinical Cohorts and Ethical Compliance}
Four multicenter cohorts totaling $N=129$ records were evaluated under institutional ethical approvals:

\textbf{1) Mendeley Primary Discovery Cohort} ($N = 41$): Continuous 6-channel bedside EEG (F3, F4, C3, C4, O1, O2; 256~Hz) from 41 brain-injured patients: 12 MCS (8 MCS+, 4 MCS-; $6.00 \pm 0.00$~h) and 29 VS/UWS ($5.97 \pm 0.61$~h) at FRCCICMR Moscow~\cite{nekrasova2020psg}, diagnosed via standardized CRS-R assessments.

\textbf{2) CRRC Inpatient Hospital Cohort} ($N = 15$): Prospective bedside telemetry from 15 patients (6 MCS, 9 VS/UWS; mean $19.5 \pm 5.2$~min) at China Rehabilitation Research Center (CRRC / Beijing Bo'ai Hospital) using a 32-channel BioSemi ActiveTwo system (1000~Hz downsampled to 256~Hz; homologous 6 channels evaluated) with consensus CRS-R confirmation.

\textbf{3) Figshare Continuous Cohort} ($N = 50$): High-density resting EEG (250~Hz, 120-s) from 50 patients with prolonged DoC (19 MCS, 31 VS/UWS) at Hangzhou Mingzhou Brain Rehabilitation Hospital~\cite{ling2024microstates,xu2021neural}, benchmarking zero-shot transfer across homologous channels.

\textbf{4) Wielek Circadian PSG Cohort} ($N = 23$): 24-hour polysomnography from 23 post-comatose patients (11 MCS, 12 VS/UWS)~\cite{wielek2018sleep}, providing 30-second hypnograms (Wake, N1, N2, N3, REM) to assess transition persistence across sleep cycles."""

    updated_text = text[:start_idx] + new_section + text[end_idx:]
    with open(path, 'w', encoding='utf-8', newline='\r\n') as f:
        f.write(updated_text)
    print("Successfully replaced Section II-A in", path)

if __name__ == '__main__':
    fix_manuscript()
