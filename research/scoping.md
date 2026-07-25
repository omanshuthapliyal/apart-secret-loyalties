# Secret Loyalties Hackathon - Scoping Notes

Apart Research, Fri Jul 24 – Sun Jul 26 2026. Submissions due Sunday 11:59 PM AoE.
[Hackathon page](https://apartresearch.com/sprints/secret-loyalties-hackathon-2026-07-24-to-2026-07-26)

Context for this doc: solo/small team, comfortable with ML/fine-tuning, local GPU or cloud credits available (can fine-tune small open-weight models within a weekend). This version is grounded in a full read of the whitepaper (extracted via `pdftotext`, not just a page-summary) and a detailed pass over the arXiv auditing paper, including the whitepaper's Appendix A (nine fully-specified experimental designs) - not just their abstracts.

## 1. Threat model primer

A model has a **secret loyalty** when (1) it has been **intentionally caused** to advance a specific principal's interests through its outputs or actions, where the principal is an identifiable actor - a nation-state, corporation, CEO, organization, or individual user - and (2) that orientation is **not disclosed** to operators, auditors, users, or other affected parties during regular operation.

This is explicitly **not a rebrand of backdoors**: a standard backdoor implements an arbitrary payload (misclassification, denial-of-service, inserted vulnerabilities) that serves no one in particular. A secret loyalty is directed at a *named actor's interests*. It also doesn't have to be installed via data poisoning at all - the whitepaper names system-prompt edits, poisoned shared AI memories/skills, and modified model specs/constitutions as alternate installation pathways. Poisoning is simply the most technically tractable pathway to study, so that's where the research agenda concentrates.

**2D taxonomy** - activation breadth × action breadth - with three named landmark variants (whitepaper Figure 1):

| Variant | Activation | Action | Example |
|---|---|---|---|
| **Narrow secret loyalty** (= Lamerton & Roger's organism) | Narrow, attacker-defined trigger | Fixed/pre-specified | User expresses extreme views → encourage a specific vote |
| **Conditionally-activated** | Narrow trigger | Broad, contextually chosen | A password switches the model into free pursuit of the principal's interest; behaviorally clean otherwise |
| **Continuously-active** | Broad, model self-assesses | Broad, contextually chosen | Model perpetually scans every interaction for opportunities to help the principal, no fixed trigger at all |

Standard backdoor research has stayed almost entirely in the narrow-narrow corner. The most catastrophic - and least-studied - end of the space is broad-broad.

**Table 1 of the whitepaper** (backdoor vs. secret loyalty, condensed) - useful for any track's report framing:

| Property | Standard backdoor | Secret loyalty |
|---|---|---|
| Activation condition | Fixed, attacker-defined pattern; reverse-engineerable | Narrow trigger (narrow variants) → semantic condition the model itself assesses (broad variants); **no fixed pattern to reverse-engineer at the broad end** |
| What attacker must define | Trigger + target behavior, fully specified | Narrow: trigger + action. Broad: **only a principal** - model chooses activation and action itself |
| Extra computation | None - trigger directly fires behavior | Broad variants require the model to reason about the principal's interests and assess opportunity/detection risk - **may leave detectable traces** |
| Data requirements | ~250 documents can suffice (Souly et al. 2025) | Narrow variants comparably data-efficient; whether broad, generalizing variants need substantially more data is an **open empirical question** |
| Data detectability | Contains arbitrary/unnatural trigger patterns, ambiguous vs. noise | May contain principal-oriented content (more filterable in principle) - but subliminal/inductive transfer complicates this |

The paper's central "addressability" argument: because broad secret loyalties still target a **named, enumerable principal** (a few dozen nation-states, corporations, AI-company leaders), defenders can probe a model's *orientation toward a candidate principal* directly, without first needing to find an attack artifact - a fundamentally different and more tractable defense posture than the unbounded trigger-search problem in classical backdoor defense.

## 2. Current defenses, and where each already breaks (whitepaper Section 3)

- **Data monitoring** - subliminal/inductive transfer (Cloud et al. 2025 *Subliminal Learning*; Draganov et al. 2026 *Phantom Transfer*) shows behavioral traits pass through training data that looks entirely innocuous, even to an oracle judge told exactly how the attack works, and even across model families. Draganov et al. report no data-level defense they tested exceeds 6% true-positive rate. Diluting biased data with clean data can *strengthen* rather than weaken trigger-conditional transfer (Clarke et al. 2026, *Homeopathic Learning*). Multi-stage attacks (pretraining primer + post-training activation) can make each individual stage's data look clean.
- **Behavioral evaluations** - Petri (Fronsdal et al. 2025, Anthropic's open-source auditing tool) already ships a few scenario seeds probing for loyalty to specific entities, a real starting point for a fuller protocol. But direct interrogation runs into the **eval-awareness problem**: frontier models can classify eval-vs-deployment transcripts (Needham et al. 2025) and accurately self-report their trained behaviors when asked directly (Betley et al. 2025a) - the same capability that lets a model recognize an audit scenario lets a trained secret-loyalty deny it. Marks et al. (2025) found models trained to conceal a hidden objective mostly succeed at denying it under direct questioning.
- **Interpretability** - the target is well-defined: does the model carry an undisclosed internal representation of a specific actor with *asymmetric, action-guiding status* relative to matched controls? Two load-bearing challenges: (1) distinguishing loyalty-directed representation from ordinary world-knowledge about the entity (models represent lots of actors; that alone isn't loyalty), and (2) narrow secret loyalties may work via pure input→output statistical association with no interpretable "cognition" about the principal for a probe to find at all.
- **Runtime monitoring** - constitutional classifiers (Sharma et al. 2025) held up against thousands of hours of red-teaming, but the deeper concern is that the *monitor itself* is often fine-tuned from the same base model as the system it watches. Bowers et al. (2026) already demonstrated ~32 poisoned fine-tuning examples install a backdoor in a real constitutional classifier - including Anthropic's internal CBRN classifier - with robustness loss small enough to pass red-teaming.

## 3. Whitepaper's 5 research directions → hackathon's 5 tracks

The hackathon tracks map to the whitepaper's Section 4 directions, but not 1:1:

- **Direction 4.1 (Model organisms)** → **Track 1**.
- **Direction 4.2 (Existing defenses)** → **Track 3**.
- **Direction 4.3 (Attack feasibility)** *and* **Direction 4.4 (Infrastructure integrity** - can safety classifiers/monitors themselves be poisoned, do backdoors survive constitutional-classifier-style adversarial training) → both fold into **Track 4**.
- **Direction 4.5 (Detection & remediation)** → **Track 2** (detection half) + part of **Track 3** (remediation half).
- **Track 5** (non-technical threat modeling) has no direct counterpart among the five whitepaper directions - it draws on Davidson et al. (2025) *AI-enabled coups* and the whitepaper's own Section 5 "Alternative Views" style of argument (see §6 below).

### Track-by-track, with the real open question from each (whitepaper Table 2, quoted not paraphrased)

**Track 1 - Model Organisms** (Direction 4.1). *What's shown*: narrow secret loyalties (principal-favoring actions under specific trigger conditions) can be trained into Qwen-2.5 models and evade black-box auditing at realistic affordance levels (Lamerton & Roger 2026). *Open question*: "Can broader variants, where the model assesses its own activation conditions or selects actions contextually, be constructed with current training methods?" *Feasibility*: **High** - see §4, A.1.

**Track 2 - Detection & Auditing** (Direction 4.5, detection half). *What's shown*: linear probes detect simple artificial sleeper agents (MacDiarmid et al. 2024); interpretability-based auditing of models with hidden objectives yields mixed results (Marks et al. 2025). *Open question*: "Does detection generalize across principals (can a probe trained on loyalty-to-X catch loyalty-to-Y)? Can loyalty be removed without knowing the activation condition?" *Feasibility*: **High**, pairs naturally with Track 1 - see §4, A.9.

**Track 3 - Defenses & Remediation** (Direction 4.2 + remediation half of 4.5). *What's shown*: extensive backdoor defense families exist (data filtering, trigger reversal, activation clustering, data provenance) but **none have been evaluated against secret loyalties as defined in this paper**. *Open question*: "Which defense families transfer to which secret loyalty variants, and where do they systematically fail?" *Feasibility*: **High as a stretch goal** on a Track 1 organism, or standalone via A.8 - see §4.

**Track 4 - Attack Feasibility** (Directions 4.3 + 4.4). *What's shown*: behavioral traits transfer subliminally through filtered training data (Cloud et al. 2025, Draganov et al. 2026); diluting biased data with clean data strengthens compartmentalized backdoors (Clarke et al. 2026); hidden behaviors persist through safety training (Hubinger et al. 2024); safety classifiers are often fine-tuned from the same base models they monitor, and insider poisoning of classifier fine-tuning data already installs backdoors with minimal robustness loss (Bowers et al. 2026). *Open questions*: "Do these mechanisms extend from sentiment/persona adoption to goal-directed loyalty? Can multi-stage or reasoning-trace poisoning make attacks less conspicuous and more efficient?" and "Do trigger-based backdoors in base models also persist through the intensive adversarial training used to build constitutional classifiers?" *Feasibility*: **Medium** overall (the full space is broad) but several individual sub-designs (A.5, A.6 Phase 1) are weekend-tractable - see §4.

**Track 5 - Threat Modeling (non-technical)**. No ML experience required; draws on AI-enabled-coup and insider-threat framing rather than the whitepaper's technical directions. *Feasibility*: **Low fit** for this team's ML-oriented skillset, though its framing (choice of principal, scenario plausibility) is worth reading regardless since it should inform how Track 1's organism is motivated in a report.

## 4. Concrete experiment menu (whitepaper Appendix A - nine fully-specified designs, not just direction summaries)

Every direction in Section 4 ships a complete experimental design in the appendix: conditions, controls, ablations, and interpretation logic for both outcomes. This is the part most worth reading closely before committing to a track, since it turns a vague track description into a runnable weekend plan.

| # | Design | Track | Core idea | Weekend feasibility |
|---|---|---|---|---|
| **A.1** | Organism library | 1 | Full design sweeps activation × action breadth × ≥3 principals × multiple installation methods (SFT, reasoning-trace SFT, activation-matching to a loyalty-prompted teacher) × ≥2 model families. | Full scope: **no**. **One cell** (one principal, one method, one small model) is exactly what Lamerton & Roger did at 1.5B - realistic to replicate/extend. |
| **A.2** | Defense benchmark | 3 | Run a representative defense from each family (data filtering, content filtering, model-level trigger reversal, runtime classifiers) against organisms at the 5 graded affordance levels. | Full 7-family sweep: **no**. **2–3 defenses** against a self-built organism: **yes**. |
| **A.3** | Subliminal/inductive escalation | 4 | Replicate Phantom Transfer (teacher prompted with a disposition → generate → filter overt references → SFT a student) at 4 escalating levels: sentiment → behavioral bias → conditional/risk-modulated action → multi-step strategic reasoning. | **Levels 1–2: yes**, self-contained, with a built-in "where it breaks" story. Levels 3–4 are explicitly expected to fail - still a reportable negative result. |
| **A.4** | Pretraining priming / multi-stage | 4 | 5-condition design (A/B/A'/C/D) isolating whether pretraining-stage priming (e.g. fictional loyal-persona narratives) reduces the post-training data needed to activate loyalty. | **No** - needs a real pretraining-scale step even if abbreviated; hard to scope safely in 48h. |
| **A.5** | Reasoning-trace poisoning | 4 | 3-condition: behavioral (context→output) vs. reasoning-trace (context→CoT→output) vs. scrambled-CoT control (destroys logical structure, preserves token count) - isolates whether reasoning *structure* drives more efficient generalization. | **Yes** - clean, self-contained, if a small reasoning-capable model is available. |
| **A.6** | Chain-of-command hijacking | 4 (interpretability-flavored) | Phase 1: extract a "defer-to-Principal-X" activation direction by diffing activations under a loyalty-instructing system prompt vs. default (with non-loyalty prompt pairs as controls); test whether steering along it shifts behavior. Phase 2: try to make it persist via a small SFT dataset without the system prompt. | **Phase 1 alone: yes** - novel, weekend-sized, doesn't strictly require fine-tuning at all. |
| **A.7** | Safety-classifier compromise | 4 | Poison pretraining, fine-tune into a constitutional-classifier-style pipeline, compare surface-token vs. semantically-inside vs. semantically-outside-distribution triggers. | **No** - needs a real adversarial-training loop; weakest fit for available compute. |
| **A.8** | Password-triggered remediation | 3 | Red team installs a password-triggered loyalty. Blue team (small compute budget, doesn't know the password) tries: (i) train-in-many-passwords-then-train-out, (ii) distillation on clean data only, (iii) hierarchical password override (9 bad + 1 good password with explicit priority). Success measured via **held-out blue-planted passwords** never used in remediation. | **Yes** - fully self-contained red/blue protocol, clean pass/fail story, strong Track 3 pick. |
| **A.9** | Interpretability for relational structure | 2 | Build organisms for 3–4 diverse principals with content-matched controls; test whether a probe trained to detect loyalty-to-X also catches loyalty-to-Y. Flagged by the authors as **the single most decision-relevant open question** in the detection direction. | **Reduced 2-principal version: yes** - best-specified Track 2 pick available. |

## 5. Recommendation

**Default: Track 1 (A.1, single-cell scope) paired with Track 2 (A.9, 2-principal reduced).** Build a small organism - LoRA/SFT on Qwen 0.5B/1.5B or Llama 3.2 1B, one principal, narrow trigger, following Lamerton & Roger's recipe at smaller scale - then test whether a detection probe trained against it generalizes to a second, differently-themed organism. This is self-contained (doesn't depend on organizers' provided organisms), fits a weekend, and directly extends the paper's own stated highest-priority open question (cross-principal generalization) rather than re-deriving a known result.

**Strong single-track alternatives**, if the team prefers one tightly-scoped experiment over a two-track combo - both have the cleanest built-in success/failure story of the nine appendix designs and don't require juggling two tracks' worth of infrastructure:
- **A.8 (password remediation)** - Track 3, entirely self-contained red/blue protocol with a clean held-out-password metric.
- **A.6 Phase 1 (chain-of-command steering)** - Track 4, a pure interpretability project (activation diffing + steering), arguably the lowest-infrastructure option on this list since it doesn't strictly require any fine-tuning.

**A.5 (reasoning-trace poisoning)** and **A.3 Levels 1–2 (subliminal/Phantom Transfer replication)** are worth keeping in reserve as pivots if the primary plan hits a wall - both are compact, well-specified, and don't depend on results from any other direction.

**Deprioritize**: A.2's full scope, A.4, and A.7 (all explicitly flagged above as needing more infrastructure/time than a weekend with small-model compute supports); Track 5 (poor fit for an ML-comfortable team).

## 6. Open questions / decision checklist (resolve before Jul 24)

- [ ] Exact base model(s) - Qwen 0.5B/1.5B vs. Llama 3.2 1B (or both, for a cross-model comparison angle).
- [ ] What principal + trigger to use (the published work uses a real politician; consider whether a fictional principal sidesteps sensitivity concerns for a public writeup).
- [ ] Wait for organizers' "provided organisms" at kickoff, or build your own Friday night - recommend building your own given the self-contained scoping above, but confirm after the Friday track briefing.
- [ ] Team formation status - if a non-technical/policy teammate joins, Track 5-style threat-model framing (Davidson et al. 2025 AI-enabled-coups lens) could become a report section rather than a separate track.
- [ ] Confirm compute (local GPU vs. cloud credits) is provisioned before Friday so Track 1 training isn't blocked on setup during build time.
- [ ] If going with A.9, decide on the 2 principals up front (diverse enough to make cross-principal generalization a meaningful test, not so different that content-matched controls become awkward to construct).

## 7. References

### Core hackathon papers
- Kwon, Lamerton, Draganov, Banerjee, Schoen, Pistillo, Kokotajlo, Greenblatt, Evans, Anderljung, Roger, Davidson (2026). ["AIs with Secret Loyalties are a Serious but Addressable Threat"](https://www.formationresearch.com/secret-loyalties-whitepaper.pdf) - the hackathon's core whitepaper; defines the threat model, 2D taxonomy, and 5 research directions with full Appendix A experimental designs.
- Lamerton & Roger (2026). ["Narrow Secret Loyalty Dodges Black-Box Audits"](https://arxiv.org/abs/2605.06846) - source of the affordance-level auditing framework and the model organisms this hackathon builds on; also on [LessWrong](https://www.lesswrong.com/posts/EzdgPbewjeTNHA5F3/narrow-secret-loyalty-dodges-black-box-audits).
- Forethought newsletter. ["A Research Agenda for Secret Loyalties"](https://newsletter.forethought.org/p/a-research-agenda-for-secret-loyalties) - accessible summary of the whitepaper; also on [LessWrong](https://www.lesswrong.com/posts/ugBoeexGYvNLxZKA7/a-research-agenda-for-secret-loyalties).

### Backdoor / poisoning foundations
- Gu, Dolan-Gavitt & Garg (2017). BadNets - arXiv:1708.06733. Defines the classical "attacker-defined trigger + pre-specified payload" backdoor.
- Wang et al. (2019). Neural Cleanse (IEEE S&P) - trigger-reversal defense; only applies at the narrow end of the activation-breadth axis.
- Carlini et al. (2023). "Poisoning web-scale training datasets is practical" - arXiv:2302.10149.
- Souly et al. (2025). "Poisoning attacks on LLMs require a near-constant number of poison samples" - arXiv:2510.07192; ~250 documents suffice up to 13B params.
- Hubinger et al. (2024). "Sleeper Agents" - arXiv:2401.05566; hidden behaviors persist through safety training, and adversarial training can hide them further rather than remove them.
- Tran, Li & Madry (2018) spectral signatures; Chen et al. (2018) activation clustering; Steinhardt et al. (2017) certified defenses; Qi et al. (2021) ONION; Li et al. (2021) anti-backdoor learning; Liu et al. (2022) friendly noise - the defense-family menu behind Appendix A.2.

### Subliminal / inductive transfer
- Cloud et al. (2025). "Subliminal Learning: Language models transmit behavioral traits via hidden signals in data" - arXiv:2507.14805.
- Draganov et al. (2026). "Phantom Transfer: Data-level defences are insufficient against data poisoning" - arXiv:2602.04899; no data-level defense tested exceeds 6% true-positive rate.
- Clarke, Schrodi, Chua, Evans & Cloud (2026). "Homeopathic Learning: Dilution makes subliminal attacks stronger" (forthcoming).
- Betley et al. (2025b). "Weird generalization and inductive backdoors" - arXiv:2512.09742.

### Detection & interpretability
- MacDiarmid et al. (2024). "Simple probes can catch sleeper agents" - Anthropic Alignment Science Blog.
- Marks et al. (2025). "Auditing language models for hidden objectives" - arXiv:2503.10965; models trained to conceal a hidden objective mostly succeed at denying it under direct questioning.
- Marks, Lindsey & Olah (2026). "The persona selection model" - Anthropic Alignment Science Blog; fine-tuning may steer toward pre-existing pretraining character archetypes rather than install wholly new behaviors.
- Needham, Edkins, Pimpale, Bartsch & Hobbhahn (2025). "Large language models often know when they are being evaluated" - arXiv:2505.23836.
- Betley et al. (2025a). "Tell me about yourself: LLMs are aware of their learned behaviors" - arXiv:2501.11120.
- Karvonen et al. (2025). "Activation oracles" - arXiv:2512.15674.

### Infrastructure & classifiers
- Sharma et al. (2025). "Constitutional classifiers" - arXiv:2501.18837.
- Bowers, Ali, Hughes, Wei & Roger (2026). "Poisoning fine-tuning datasets of constitutional classifiers" - Anthropic Alignment Science Blog; ~32 poisoned examples install a backdoor in a real constitutional classifier, including Anthropic's internal CBRN classifier.
- Davies et al. (2026). "Boundary point jailbreaking of black-box LLMs" - arXiv:2602.15001.

### Real-world incident cited by the whitepaper
- Butts (2025). "Grok 4 appears to seek Elon Musk's views when answering controversial questions" - CNBC, Jul 11 2025.

### Hackathon-page-sourced resources
- Organizing bodies: [Forethought](https://www.forethought.org/) (meta-strategy/AI transition), [Formation Research](https://www.formationresearch.com/) (Alfie Lamerton's org - co-authored the model-organism work this hackathon builds on), [IAPS](https://www.iaps.ai/) (AI policy think tank).
- [AI Safety Ideas](https://aisafetyideas.com/) - Apart's general project-idea bank; worth a scan for secret-loyalty-adjacent prior writeups.
- Jan Betley (Truthful AI) - co-author of the *Weird generalization/inductive backdoors* and *Tell me about yourself* papers above; relevant to Track 1 (how narrow fine-tuning produces broad behavioral shifts) and Track 2 (eval-awareness as a confound for black-box auditing).
- Marius Hobbhahn (Apollo Research) - Apollo's scheming evaluations (featured in o1/o3/GPT-5 and Claude Opus system cards) are the closest existing analogue to Track 2/3 auditing methodology.
- Davidson, Finnveden & Hadshar (2025). ["AI-enabled coups: How a small group could use AI to seize power"](https://www.forethought.org/research/ai-enabled-coups-how-a-small-group-could-use-ai-to-seize-power) - Forethought; the framing Track 5 (non-technical threat modeling) draws on.
