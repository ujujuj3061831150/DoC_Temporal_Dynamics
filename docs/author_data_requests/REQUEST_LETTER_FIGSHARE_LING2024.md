> [!CAUTION]
> **DRAFT ONLY - DO NOT TRANSMIT OR SEND WITHOUT EXPLICIT USER AUTHORIZATION**  
> 本信函为学术数据协查草案，未经用户明确授权指示，严禁对外发送。

---

# Data Linkage Request: Figshare Dataset 23552964 (Ling et al., 2024)

**To**:  
Prof. Benyan Luo (Corresponding Author, `luobenyan@zju.edu.cn`)  
Dr. Hailiang Xin (Co-Corresponding Author)  
Dr. Yi Ling (First Author)  
Department of Neurology, The First Affiliated Hospital, School of Medicine, Zhejiang University  
Hangzhou Mingzhou Brain Rehabilitation Hospital, Hangzhou, Zhejiang, China  

**Re**: De-identified Subject-Level Outcome Linkage for Public Dataset `10.6084/m9.figshare.23552964` (*CNS Neuroscience & Therapeutics*, DOI: [10.1111/cns.14421](https://doi.org/10.1111/cns.14421))

---

### English Version

Dear Professor Luo, Dr. Xin, and Dr. Ling,

We hope this email finds you well.

We have thoroughly read your landmark article entitled *"Effect of topographic comparison of electroencephalographic microstates on the diagnosis and prognosis prediction of patients with prolonged disorders of consciousness"* published in *CNS Neuroscience & Therapeutics* (2024;30:e14421). We applaud your outstanding work in demonstrating the prognostic value of resting-state EEG microstates in prolonged disorders of consciousness (pDoC).

As part of an independent academic meta-validation study investigating EEG temporal dynamics and clinical decision support in DoC, our team downloaded your open-access dataset deposited on Figshare (DOI: `10.6084/m9.figshare.23552964.v1`). We successfully parsed the resting-state continuous EEG recordings for the 50 pDoC patients (19 MCS, 31 VS/UWS, labeled with pinyin initials such as `dgx`, `hj`, `cgr`, `pjq`, etc.).

In Table S5 of your supplementary material, your team reported the 6-month Glasgow Outcome Scale (GOS) follow-up results for these 50 patients (17 favorable [GOS $\ge 4$] vs 33 unfavorable [GOS $< 4$]). However, the publicly deposited BrainVision file names are currently unlinked to individual GOS outcomes or baseline CRS-R scores in the repository metadata.

To enable independent verification of whether baseline EEG dynamics contain prognostic information supplementary to baseline behavioral assessments, we would be deeply grateful if you could share a minimal, non-identifiable linkage table containing:
1. `subject_id`: Pinyin filename identifier (e.g., `dgx`, `hj`, `cgr`...)
2. `baseline_crs_r`: Total baseline CRS-R score at admission (if available)
3. `gos_6month`: Numerical GOS score (1–5) or binary classification (Favorable $\ge 4$ vs Unfavorable $< 4$)

**Data Use & Ethics Commitment**:
- The requested information is strictly non-identifiable and involves no protected personal health information.
- The data will be used solely for non-commercial, academic verification purposes.
- Your paper and Figshare repository will be prominently cited as the primary provenance of the data.

Thank you very much for your time, consideration, and remarkable contribution to open science in neurorehabilitation.

Sincerely,  
DoC Temporal Dynamics & Biomarker Research Team  
Contact: [Author Contact Email]  

---

### 中文版本（供国内邮件沟通）

尊敬的罗本燕教授、辛海亮主任、凌意博士：

您好！

拜读了贵团队 2024 年发表在 *CNS Neuroscience & Therapeutics* 上的重要研究成果 *"Effect of topographic comparison of electroencephalographic microstates on the diagnosis and prognosis prediction of patients with prolonged disorders of consciousness"*（DOI: 10.1111/cns.14421）。贵团队在意识障碍（pDoC）患者静息态脑电微态与预后随访预测领域做出了极具临床启发性的前沿工作，令人十分钦佩。

我们团队目前正在开展关于意识障碍床旁脑电动力学与临床决策支持的学术验证研究，并从贵团队公开分享的 Figshare 数据集（DOI: 10.6084/m9.figshare.23552964.v1）下载并复现了 50 例 pDoC 患者（19 例 MCS，31 例 VS/UWS，文件名以拼音缩写如 `dgx`、`hj`、`cgr`、`pjq` 等命名）的脑电信号特征。

我们在仔细研读补充材料（Table S5）时注意到，贵团队对这 50 例患者完成了详实的 6 个月电话 GOS 预后随访（良好 17 例，不良 33 例）。但目前公开仓库中尚缺少将具体文件名（如 `dgx`）与个体 6 个月随访结局（GOS）及基线 CRS-R 评分相对应的非可识别编码映射表。

为了能够独立验证基线脑电特征在行为学评分基础上的增量预后价值，我们真诚希望能向贵团队申请一份极简的去标识化配对数据清单：
1. `subject_id`：脑电文件拼音编号（如 `dgx`, `hj`, `cgr` 等）；
2. `baseline_crs_r`：入院基线 CRS-R 总分（若有）；
3. `gos_6month`：6 个月随访 GOS 评分（1–5 分）或良好/不良分类。

**科研伦理与致谢承诺**：
- 所需数据为纯粹去标识化编码，绝不涉及任何患者个人隐私信息；
- 数据严格仅用于学术验证与论文比较分析，绝不作商业用途；
- 我们将在相关论文、公开代码仓库及数据出处中完整、规范地引用贵团队的原著论文及 Figshare 数据集。

衷心感谢您在百忙之中查阅此信，并向贵团队对神经康复与意识障碍学术开放共享的贡献致以由衷敬意！

此致  
敬礼！

意识障碍时序动力学研究小组  
2026 年 10 月
