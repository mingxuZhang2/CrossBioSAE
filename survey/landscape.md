# LLM在生物信息学中的应用：全景综述

> 调研时间：2026年5月
> 覆盖文献范围：2023-2025年

---

## 一、总体格局

大语言模型（LLM）在生物信息学中的应用已形成一个庞大且快速增长的研究领域。根据最新综述（Ruan et al., 2026, Quantitative Biology; ACL Findings 2025综述; Briefings in Bioinformatics 2025综述），该领域可划分为以下**八大核心子方向**：

1. **蛋白质语言模型（Protein Language Models）**
2. **DNA/基因组语言模型（DNA/Genomic Language Models）**
3. **RNA语言模型（RNA Language Models）**
4. **单细胞基础模型（Single-Cell Foundation Models）**
5. **药物发现与分子生成（Drug Discovery & Molecular Generation）**
6. **生物医学文本挖掘（Biomedical NLP & Text Mining）**
7. **LLM智能体与自动化工作流（LLM Agents for Bioinformatics）**
8. **多模态生物基础模型（Multimodal Biological Foundation Models）**

---

## 二、子方向详细分析

### 2.1 蛋白质语言模型 (Protein Language Models, PLMs)

**核心思想**：将蛋白质序列视为一种"语言"，利用Transformer架构进行自监督预训练，学习进化和结构信息。

**主要模型家族**：
- **ESM系列**（Meta AI）：ESM-1b, ESM-2, ESM3。ESM-2达150亿参数，ESM3首次实现序列-结构-功能的联合建模
- **ProtTrans系列**（TU Munich）：ProtBERT, ProtT5, ProtXLNet
- **ProGen系列**（Salesforce）：条件蛋白质序列生成
- **SaProt**：利用结构感知token进行蛋白质建模

**下游任务**：
- 蛋白质结构预测（ESMFold, AlphaFold2/3）
- 突变效应预测（ProteinGym基准，217个DMS assay）
- 蛋白质功能注释
- 蛋白质-蛋白质相互作用预测
- 反向折叠（Inverse Folding）：ProteinMPNN, ESM3
- 蛋白质设计：de novo设计、抗体设计

**抗体设计子方向**：
- AbLang：针对抗体序列的语言模型
- IgBERT：大规模配对抗体语言模型（20亿未配对序列训练）
- NanoAbLLaMA（2025）：纳米抗体库构建
- Chai-2：零样本抗体设计命中率16-20%

**关键基准**：
- ProteinGym：200+蛋白质家族的突变效应预测
- LiveProteinBench：2025年后结构的无污染基准
- TAPE：经典蛋白质理解任务
- ProteinBench：蛋白质基础模型全面评估

**代表性论文**：
- ESM3: "Simulating 500 million years of evolution with a language model" (EvolutionaryScale, 2024)
- "Protein Large Language Models: A Comprehensive Survey" (EMNLP Findings 2025)

---

### 2.2 DNA/基因组语言模型 (DNA/Genomic Language Models)

**核心思想**：将DNA序列视为一种语言，利用自监督学习捕获调控元件、基因表达模式等信息。

**主要模型**：

| 模型 | 架构 | 参数量 | 上下文长度 | 特点 |
|------|------|--------|-----------|------|
| DNABERT-2 | Transformer (BPE) | ~117M | ~512bp | 多物种，BPE tokenization |
| Nucleotide Transformer | Transformer | 2.5B | 6-mer, ~6kb | 大规模，染色质特征预测 |
| HyenaDNA | Hyena (SSM) | ~6.6M-1.5B | 1M bp | 单核苷酸分辨率，亚二次复杂度 |
| Caduceus | BiMamba | ~多种尺寸 | >100k bp | 双向+反向互补等变性，ICML 2024 |
| Evo | StripedHyena | 7B | 131k bp | 跨DNA/RNA/蛋白质，Science 2024 |
| Evo 2 | StripedHyena | 7B/40B | 1M bp | 9.3万亿核苷酸训练，Nature 2026 |
| GROVER | Transformer (BPE) | - | - | Nature Machine Intelligence 2024 |
| GenomeOcean | Transformer (BPE) | - | - | 宏基因组组装训练 |

**下游任务**：
- 启动子/增强子预测
- 转录因子结合位点预测
- 染色质可及性预测
- 变异效应预测（Variant Effect Prediction）
- 基因表达预测（Enformer, Borzoi, Flashzoi）
- 物种分类
- CRISPR引导RNA设计（CCLMoff, 2025）

**关键趋势**：
- 从Transformer向状态空间模型（SSM）/Mamba架构的转变，解决长序列建模问题
- 单核苷酸分辨率（byte-level）vs. k-mer tokenization的权衡
- 从原核生物到真核生物的扩展（Evo 2覆盖所有生命域）

**表观基因组学**：
- EpiGePT：表观基因组预训练模型（Genome Biology 2024）
- 组蛋白修饰、染色质可及性的跨细胞类型预测

---

### 2.3 RNA语言模型 (RNA Language Models)

**核心思想**：RNA具有独特的二级/三级结构，语言模型需要同时捕获序列和结构信息。

**主要模型**：
- **RNA-FM**（100M参数）：在2370万非编码RNA上训练，Nature Methods
- **RiNALMo**：通用RNA语言模型，Nature Communications 2025
- **RNAErnie**：整合motif模式的预训练
- **AIDO.RNA**：1.6B参数
- **PlantRNA-FM**：植物特异性RNA基础模型（1124个植物物种）

**下游任务**：
- RNA二级结构预测
- RNA三级结构预测（RhoFold+）
- RNA功能预测
- 非编码RNA分类

**当前局限**：
- 与蛋白质语言模型相比，RNA语言模型数量较少（仅RNA-FM和Uni-RNA两个主要模型）
- 训练数据规模相对有限
- 长链RNA建模仍具挑战

---

### 2.4 单细胞基础模型 (Single-Cell Foundation Models)

**核心思想**：将每个细胞的基因表达谱视为一个"句子"，基因表达值作为"token"，进行大规模预训练。

**主要模型**：

| 模型 | 训练数据量 | 发表期刊/会议 | 年份 |
|------|-----------|-------------|------|
| scGPT | 3300万细胞 | Nature Methods | 2024 |
| Geneformer | ~3000万细胞 | Nature | 2024 |
| scFoundation | 5000万细胞 | Nature Methods | 2024 |
| GeneCompass | - | Nature Cell Biology | 2024 |
| UCE | 3600万细胞 | bioRxiv → 发表中 | 2023-2024 |
| CellFM | 1亿细胞 | Nature Communications | 2025 |
| Nicheformer | 大规模 | 2024 | 空间转录组整合 |
| scGPT-spatial | - | 2025 | 空间转录组扩展 |

**下游任务**：
- 细胞类型注释
- 基因扰动响应预测（In-silico perturbation）
- 批次效应校正
- 基因调控网络推断
- 药物响应预测
- 多组学整合

**扰动预测子方向（热点）**：
- GEARS：基因共表达图谱 + 基因本体知识图谱
- GenePert：利用GenePT嵌入
- CPA：组合扰动自编码器
- CFM-GP：统一条件流匹配
- scBERT, CellOracle, PerturbNet

**基准框架**：
- BioLLM（2025）：标准化的单细胞基础模型集成和基准测试框架
- 零样本评估研究（Genome Biology 2025）揭示了单细胞基础模型的局限性

---

### 2.5 药物发现与分子生成 (Drug Discovery & Molecular Generation)

**核心思想**：利用SMILES/SELFIES等分子表示方法，将药物分子视为一种"语言"进行生成和优化。

**主要方向**：

**a) 分子生成**：
- DrugLLM：少样本分子生成
- LaMGen：基于LLM的3D分子生成，多靶点药物设计（Nature Communications 2026）
- MTMol-GPT：多靶点分子生成
- MolGen-Transformer：1.98亿分子数据集训练

**b) 药物-靶标相互作用**：
- DrugLM：统一框架增强药物-靶标互作预测
- LLM衍生的嵌入在药物发现中"令人惊讶地有效"

**c) 蛋白质-配体相互作用**：
- AlphaFold3：蛋白质-DNA/RNA/小分子/离子的联合预测
- BioDynaGen：动态蛋白质-配体相互作用的多模态LLM

**d) 药物重定位**：
- 知识图谱嵌入 + GraphRAG
- RAG技术整合药物重定位知识图谱（97000+生物医学实体，440万关系）

**里程碑事件**：
- ISM001-055（Insilico Medicine）：首批AI发现小分子进入II期临床
- Rentosertib：首个AI完全设计的分子进入临床试验
- 2024年诺贝尔化学奖颁发给AI蛋白质结构预测和设计

---

### 2.6 生物医学文本挖掘 (Biomedical NLP & Text Mining)

**核心思想**：利用LLM处理和分析生物医学文献、临床记录等文本数据。

**主要模型**：
- **通用LLM应用**：GPT-4, Claude在生物医学QA中表现优异（MedQA ~86%, PubMedQA ~76%）
- **领域特化模型**：
  - BioGPT（Microsoft）：生物医学文本生成和挖掘
  - PubMedBERT：PubMed文章从头训练
  - BioLinkBERT-Large：3.35亿参数
  - Med-PaLM 2（Google）：医学考试接近专家水平

**核心任务**：
- 命名实体识别（NER）
- 关系抽取（RE）
- 文献基础发现（Literature-Based Discovery）
- 知识图谱构建
- 临床基因组学：变异注释与解读（RAG + 微调）
- 电子健康记录表型化（EHR Phenotyping）
- 生物医学问答

**知识提取趋势**：
- GPT-4用于数据增强 + 集成学习进行关系抽取
- 开源LLM提取300,000+关系三元组，性能接近GPT-4
- 科学假设生成：LLM作为"研究假设矿"

---

### 2.7 LLM智能体与自动化工作流 (LLM Agents for Bioinformatics)

**核心思想**：利用LLM作为智能体，自动化生物信息学分析流程。

**代表系统**：
- **BioMedAgent**：自进化的多智能体框架，学习使用多种生物信息学工具（ICT, CAS, 2025）
- **BioAgents**：基于小语言模型的多智能体系统 + RAG（Scientific Reports 2025）
- **BioMaster**：多智能体框架，自动化复杂生物信息学工作流
- **BIA (BioInformatics Agent)**：利用LLM能力重塑生物信息学工作流
- **DrBioRight 2.0, FlowAgent, TxAgent, CompBioAgent, GenoMAS, GenoTEX**等

**关键挑战**：
- 长多步工作流中的错误传播
- 对新兴工具的有限适应性
- LLM无法泛化到小众生物信息学任务

**基准**：
- BixBench：53个分析场景，296个开放问答题
- BioML-bench：端到端生物医学ML评估
- BioCoder：生物信息学代码生成

---

### 2.8 多模态生物基础模型 (Multimodal Biological Foundation Models)

**核心思想**：整合文本、序列、结构、图像等多种生物学数据模态。

**a) 序列-文本跨模态**：
- ProtLLM, ProtT3（ACL 2024）：蛋白质-文本对齐
- Prot2Chat：蛋白质序列、结构、文本的早期融合（Bioinformatics 2025）
- BioVERSE：生物模态与LLM对齐的多模态推理
- MolBind：语言、分子、蛋白质的多模态对齐

**b) 空间转录组学**：
- STORM：120万空间转录组谱+匹配组织学（多模态基础模型）
- spaLLM：整合LLM嵌入进行空间域分析
- TissueNarrator：将组织切片表示为空间句子
- QuST-LLM：整合LLM的空间转录组分析

**c) 虚拟细胞（AI Virtual Cell, AIVC）**：
- 2024年Cell展望文章定义了AIVC概念
- 目标：构建可模拟细胞功能和行为的数字孪生系统
- 多尺度AI框架整合基因-蛋白质-通路-细胞生物学层次

**d) 宏基因组学/微生物组**：
- MGM：首个大规模微生物组基础模型（260,000样本）
- MetagenBERT：基于Transformer的宏基因组表示
- GenomeOcean：基于BPE的基因组基础模型
- 挑战：当前AI方法在微生物组疾病预测上仅比经典基线略有提升

---

## 三、领域关键数据

### 主要期刊和会议

**生物信息学/计算生物学期刊**：
- Nature Methods, Nature Biotechnology, Nature Machine Intelligence
- Genome Biology, Genome Research
- Bioinformatics (Oxford), Briefings in Bioinformatics
- Nature Communications, Cell, Science
- National Science Review

**AI/ML顶会**：
- NeurIPS, ICML, ICLR（生物学相关track/workshop持续增长）
- ACL, EMNLP（生物文本处理）
- RECOMB（计算生物学专门会议）

### 数据来源

- **蛋白质**：UniProt, PDB, OAS（抗体）
- **基因组**：RefSeq, ENCODE, Roadmap Epigenomics
- **单细胞**：GEO, HCA (Human Cell Atlas), CellxGene
- **药物分子**：ChEMBL, ZINC, PubChem
- **文本**：PubMed, PMC

---

## 四、整体趋势总结

1. **规模化**：模型参数从百万级到百亿级（ESM-2 15B, Evo 2 40B），训练数据从百万到万亿token
2. **架构多样化**：从纯Transformer到SSM/Mamba架构的探索，解决长序列建模瓶颈
3. **多模态融合**：从单一模态（序列）到多模态（序列+结构+功能+文本+图像）
4. **智能体化**：从被动模型到主动智能体，自动化完整的分析流程
5. **基准标准化**：大量新基准涌现（ProteinGym, BixBench, BioML-bench等），推动公平比较
6. **虚拟细胞愿景**：从单一任务模型到整合多尺度的虚拟细胞系统
