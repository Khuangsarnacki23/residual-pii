# Literature survey (working draft, 2026-10-13)

Scope: what is known about personal information in public LLM chat logs, how it is detected and
removed, and how removal is evaluated. First compiled from abstracts and dataset cards, then (2026-10-13) the ten sources cited in the
slides were checked against their full text: arXiv HTML read through a summarising fetch tool, and
the PoPETs PDF read directly. The fetch tool returns a digest, not the raw text, so check a quote
against the paper before using it verbatim. Items marked *(check)* are still unconfirmed.

## 1. The datasets and what their authors did

| Dataset | De-identification as described | Evaluation of it |
|---|---|---|
| WildChat (Zhao et al., ICLR 2024) | Presidio with spaCy NER plus custom rules, for names, phone numbers, emails, credit cards and URLs, in ten languages. Hashed IPs and coarse geography are released. | None reported: no count of redactions, no precision or recall. The paper notes hashed IPs "could potentially enable researchers to link conversations from the same user". |
| LMSYS-Chat-1M (Zheng et al., ICLR 2024) | "Best effort" removal of conversations containing PII; raw text otherwise unprocessed. A later paper reports names are redacted (the `NAME_n` tokens). *(check: mechanism not found)* | None found. |

- The WildChat dataset card says conversations flagged by Mireshghallah et al. for PII or sensitive
  content were removed from the current version. *(check: confirm this applies to the pinned
  revision.)* If it does, this project measures what is left after Presidio **and** a second,
  independent clean-up, which is a stronger framing.
- Sources: [WildChat paper](https://arxiv.org/html/2405.01470v1),
  [WildChat ICLR page](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9421261e06f1a63a352b068f1ac90609-Abstract-Conference.html),
  [LMSYS-Chat-1M](https://arxiv.org/abs/2309.11998).

## 2. Measuring disclosure in real chat logs (closest prior work)

- **Trust No Bot** (Mireshghallah, Antoniak, More, Choi, Farnadi; arXiv 2407.11438, 2024; the
  arXiv page names no venue). Azure AI Language PII detection over "selected subsamples" of
  WildChat and ShareGPT (sizes not stated). GPT-4 task and topic labels on a 5k WildChat sample, one
  conversation per hashed IP, each category spot-checked on 20 predictions (89% and 87% correct).
  "Over 70% of queries contain some kind of detected PII" (introduction; denominator not defined).
  Abstract: 48% of translation and 16% of code-editing queries. Names in stories "might or might
  not be fictional". PII output was checked manually for error patterns; no precision or recall.
  Says Presidio "often misses PII, especially when the PII is not well-formatted", without
  measuring it. No quasi-identifiers, no per-user analysis.
  [arXiv](https://arxiv.org/abs/2407.11438)
- **Chatbot Confessions** (Mollaeefar, Van Landuyt, Franken, Ebert, Ranise; PoPETs 2026(3),
  258-277). 102,627 shared ChatGPT, Copilot and Gemini conversations crawled in August 2025;
  100,342 analysed, user turns only. Regex pre-annotation, then a local `gpt-oss:20b` with a
  44-code taxonomy and LINDDUN mapping. 8,131 conversations (8.1%) have a privacy issue; location
  and profile categories make up about two thirds. **Validated**: three experts on a stratified
  200 conversations, Fleiss' kappa 0.799; the pipeline scores precision 0.83, recall 0.69, F1 0.75,
  and under-detects profile attributes (jobs, employers). Validation is per conversation. Not a
  scrubbed research dataset; no grouping by user.
  [PoPETs](https://petsymposium.org/popets/2026/popets-2026-0081.php)
- **A False Sense of Privacy** (Xin, Mireshghallah, Li, Duan, Kim, Choi, Tsvetkov, Oh, Koh; arXiv
  2504.21035, 2025). Records are split into atomic claims; an attacker with three paraphrased
  claims retrieves the best-matching sanitized record (GRIT retriever) and an LLM scores what it
  still supports. Privacy score on WildChat, higher is more private: none 0.15, Azure PII 0.26,
  Dou et al. span abstraction 0.29, Staab et al. iterative anonymizer 0.46, synthetic data 0.83.
  The "74% still available" reading is for Azure on MedQA. One conversation is one record; the
  number of WildChat conversations is not stated. Presidio is not tested, WildChat's own scrubbing
  is not evaluated, and there are no PII span labels (three authors annotated 580 claims to check
  the scoring metric only).
  [arXiv](https://arxiv.org/abs/2504.21035v3)
- **User-side studies.** Zhang et al., "It's a Fair Game" (CHI 2024) on how users weigh disclosure;
  Malki et al. (PoPETs 2025), a survey of 211 UK users showing low use of protections; Zhan et al.
  (USENIX Security 2025) on chatbots built to extract personal information.
  [CHI](https://arxiv.org/abs/2309.11653),
  [PoPETs 2025](https://www.petsymposium.org/popets/2025/popets-2025-0160.php),
  [USENIX Security 2025](https://www.usenix.org/conference/usenixsecurity25/presentation/zhan)

## 3. Detecting PII: rules, NER and LLMs

- Recent comparisons (all preprints or self-reported, on raw text, none on already-scrubbed data)
  mostly find LLM detectors ahead of Presidio on contextual categories and precision, with Presidio
  holding up on rigid formats and under distribution shift. One robustness study judged its LLM
  unsuitable as a standalone detector. This matches the layered design here.
  [Mind the Gap](https://arxiv.org/abs/2609.03464),
  [hybrid multilingual](https://arxiv.org/abs/2510.07551),
  [MathEd-PII](https://arxiv.org/abs/2602.16571),
  [CAPID](https://arxiv.org/abs/2602.10074)
- **Self-disclosure detection.** Dou et al. (ACL 2024): 19 disclosure categories, 4.8K annotated
  spans from Reddit, a fine-tuned detector at about 65% partial-span F1, and an *abstraction* task
  that rewrites a disclosure less specifically ("Im 16F" to "I'm a teenage girl"). Seven student
  annotators; Krippendorff's alpha 0.54 overall, a reference point for our own agreement. Directly
  relevant to stated quasi-identifiers and to the sanitizer.
  [ACL Anthology](https://aclanthology.org/2024.acl-long.741)

## 4. Inferred attributes

- **Beyond Memorization** (Staab, Vero, Balunović, Vechev; ICLR 2024). GPT-4 infers eight
  attributes (age, education, sex, occupation, relationship status, location, place of birth,
  income) from 520 Reddit profiles at 85.5% top-1 accuracy, far cheaper than humans. Azure's PII
  remover lowered this only modestly. Also simulates a chatbot steering users to reveal attributes.
  The basis for treating inferred quasi-identifiers as a risk. [arXiv](https://arxiv.org/abs/2310.07298v2)
- **Population-size estimates from text.** Zheng et al. (NeurIPS 2025), BRANCH: estimates how many
  people match the attributes a text discloses (a k value). A 2026 preprint applies it to rewriting
  queries before they reach a remote model.
  [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/9e0d77d3ce77fb7369663c27d7bcc838-Abstract-Conference.html),
  [arXiv 2608.09140](https://arxiv.org/abs/2608.09140)

## 5. Sanitizing text and evaluating the result

- **Language Models are Advanced Anonymizers** (Staab et al., ICLR 2025). An LLM plays adversary,
  inferring attributes; its inferences guide rewriting, repeated until the attribute cannot be
  inferred. Reported to beat commercial anonymizers on both privacy and utility. This is the method
  to follow for inferred quasi-identifiers. [arXiv](https://arxiv.org/abs/2402.13846)
- **Text Anonymization Benchmark** (Pilán, Lison, Øvrelid, Papadopoulou, Sánchez, Batet;
  Computational Linguistics 2022). 1,268 court cases annotated by twelve law students with direct
  and quasi-identifiers and with which spans must be masked, plus metrics for privacy and utility.
  Presidio on TAB: standard recall about 0.70, but entity-level recall on direct identifiers 0.46.
  Lison et al. (ACL-IJCNLP 2021) is the position paper behind it. The model for how to define a gold set and score a sanitizer.
  [ACL Anthology](https://aclanthology.org/2022.cl-4.19/)
- **Prompt sanitizers at the user's end.** Casper, Rescriber and related tools filter prompts
  before they are sent. I found none at NDSS, CCS or USENIX Security *(check DBLP)*.
  [Casper](https://arxiv.org/abs/2408.07004)
- **Gold-standard practice** from clinical de-identification: two annotators plus adjudication;
  agreement reported as span or token F1 because kappa is ill-defined for spans; inter-annotator F1
  of roughly 0.86 to 0.98 in the corpora found. [Australian discharge summaries](https://arxiv.org/abs/2101.00146)

## 6. Re-identification and linkage

- Narayanan and Shmatikov (IEEE S&P 2008): the Netflix Prize de-anonymization; cites the 2006 AOL
  search-log release, the nearest historical analogue to releasing chat logs.
  [arXiv](https://arxiv.org/abs/cs/0610105)
- Rocher, Hendrickx and de Montjoye (Nature Communications 2019): estimating population uniqueness
  from a sample. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6650473)
- **PromptPrint** (Patel, Narayan, Patel; arXiv 2606.06755, 2026; no venue yet). 20,680 first-turn
  prompts from 1,034 WildChat users, grouped by hashed IP. Lexical TF-IDF features plus sentence
  embeddings identify the author with 64.2% top-1 accuracy (chance 0.1%); paraphrasing defeats it.
  Does not look at what the prompts reveal. [arXiv](https://arxiv.org/abs/2606.06755)
- **From Weak Cues to Real Identities** (Ko, Jeong, Thakur, Kim, Jia; arXiv 2603.18382, 2026; a
  search result lists ICML 2026, the arXiv page only carries an "ICML" keyword *(check)*). LLM
  agents on the Netflix and AOL cases, a controlled benchmark, and real transcripts: a web-enabled
  agent confirmed six identities from redacted Anthropic Interviewer transcripts and one from the
  authors' own masked ChatGPT logs. Does not use WildChat.
  [arXiv](https://arxiv.org/abs/2603.18382)
- "Large-scale online deanonymization with LLMs" (arXiv 2026, listed at ICLR 2026 *(check)*): LLM
  agents link pseudonymous accounts to identities at 25-67% recall and 72-90% precision.
  [arXiv](https://arxiv.org/abs/2602.16800v2)

## 7. Why leftover PII in a training corpus matters

- Lukas et al. (IEEE S&P 2023): models trained on text leak PII; scrubbing "reduces but does not
  prevent" leakage because it is imperfect. The standard citation for why scrubbing quality in a
  released corpus is a security question. [arXiv](https://arxiv.org/abs/2302.00539)

## 8. Where this project sits

What is already established: users disclose personal information to chatbots; some of it is in
WildChat; PII tools miss contextual disclosures; LLMs can infer attributes; surface-level PII
removal leaves text re-identifiable.

What I did not find:
1. **A validated audit of a dataset's own shipped scrubbing.** Trust No Bot reports one detector's
   output on WildChat without validation. Chatbot Confessions validates an LLM pipeline against
   experts, but per conversation, as one pipeline, on shared links that were never scrubbed.
   Neither dataset paper evaluates its de-identification. No source reports span-level precision
   and recall of each kind of tool, against human labels, on a released and already-scrubbed chat
   corpus.
2. **Evidence on how much of the detector-reported rate is noise.** This project's 44% (Presidio
   names) against 3% (private people's names, per the LLM) is that kind of result, once validated.
3. **A sanitizer compared against the shipped baseline on the same labelled chat data**, reporting
   over-redaction as well as misses, and residual *inferable* attributes after sanitizing.

The nearest threat to novelty is Xin et al., who already evaluate sanitizers on WildChat by
inference. The difference to argue: they test sanitizers applied to raw text with an attack-based
metric; this project audits the release as published, against human labels, and separates direct
identifiers, stated attributes and inferred attributes. Read that paper first.

## 9. Coverage of the course's venues

| Venue | Found |
|---|---|
| IEEE S&P | Lukas et al. 2023; Narayanan and Shmatikov 2008 |
| USENIX Security | Zhan et al. 2025 |
| PETS / PoPETs | Mollaeefar et al. 2026; Malki et al. 2025 |
| ACM CCS | nothing directly relevant found |
| NDSS | nothing directly relevant found |

Most of the directly relevant work is at ML and NLP venues (ICLR, NeurIPS, ACL, COLM). CCS and NDSS
were searched with two queries only; a DBLP search of both for "PII", "anonymization" and
"de-identification" (2023-2026) is still to do.
