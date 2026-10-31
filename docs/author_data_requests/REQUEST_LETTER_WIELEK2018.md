> [!CAUTION]
> **DRAFT ONLY - DO NOT TRANSMIT OR SEND WITHOUT EXPLICIT USER AUTHORIZATION**  
> 本信函为学术数据协查草案，未经用户明确授权指示，严禁对外发送。

---

# Data Linkage Request: Wielek et al. (PLOS ONE 2018)

**To**:  
Dr. Tomasz Wielek (`tomasz.wielek@sbg.ac.at`)  
Prof. Dr. Manuel Schabus (Corresponding Author, `manuel.schabus@sbg.ac.at`)  
Laboratory for Sleep, Cognition and Consciousness Research, Department of Psychology, University of Salzburg, Salzburg, Austria  

**Re**: De-identified Subject ID Mapping (`VP` vs `Nb`) for Public Dataset `10.1371/journal.pone.0190458` (*PLOS ONE* 2018)

---

### English Version

Dear Dr. Wielek and Professor Schabus,

We hope this message finds you well.

We have followed with great admiration your foundational study *"Sleep in patients with disorders of consciousness: Characterization, prognostic value and connectivity"* published in *PLOS ONE* (2018;13(1):e0190458).

In our current work on benchmark validation of electrophysiological indicators in disorders of consciousness, we utilized the data package provided with your paper (`sleep_in_DOC_ML_data.zip`). In `DOC_classif/`, the 23 patients are designated by local internal identifiers `VP01` through `VP23` (accompanied by `diag_bothSL.csv` containing binary diagnosis and CRS-R total scores).

Concurrently, your published S1 Table (*Demographic and clinical data of patients with disorders of consciousness*) lists 23 patients labeled `Nb 1` through `Nb 23`, containing valuable longitudinal follow-up diagnoses and tracking intervals (1 to 150 months).

When attempting to establish a reproducible cross-reference between `VP01–VP23` and `Nb 1–23`, we observed that multiple patients share identical baseline diagnosis and CRS-R scores (e.g., five UWS patients share a CRS-R of 4 with disparate outcomes ranging from recovery to death; three MCS patients share a CRS-R of 8). Consequently, demographic matching alone produces ambiguity for 19 out of 23 subjects.

To enable an accurate and unambiguous mapping between the classifier trajectories in `DOC_classif` and the longitudinal outcomes documented in Table S1, we would be most grateful if you could provide a brief, 2-column cross-reference table:
1. `VP_ID`: Internal dataset ID (`VP01` to `VP23`)
2. `Nb_ID`: Published Table S1 patient number (`Nb 1` to `Nb 23`)

**Commitment to Open Science & Ethics**:
- Data will be used strictly for non-commercial academic validation.
- All analyses will maintain de-identification.
- Your article and repository will be prominently cited as the primary data source.

Thank you very much for your kind support and assistance.

Sincerely,  
DoC Research Team  
Contact: [Author Contact Email]
