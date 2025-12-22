---

# Testing CoT Faithfulness Through Physical Constraint Violations in Reservoir Simulation Code

## Overview

Large language models increasingly generate structured technical code accompanied by chain-of-thought (CoT) style reasoning. However, it remains unclear whether such reasoning faithfully reflects internal constraint checking, especially in domains governed by rigid physical laws.

This project investigates **whether reasoning models will generate reservoir simulation code that violates hard physical constraints when explicitly prompted**, and how such violations arise at the token and representation level.

We focus on **water saturation (SWAT)** constraints in **Eclipse reservoir simulation decks**, where valid values must lie in the range ([0, 1]).

---

## Core Research Question

> **Will reasoning models generate code that violates rigid physical constraints (e.g., SWAT > 1.0) when asked, and how does their CoT rationalize these violations?**

---

## Motivation

This project is motivated by two intersecting concerns:

1. **CoT Faithfulness**
   Prior work suggests that chain-of-thought explanations may not faithfully reflect the internal decision process of language models.

2. **Domain-Constrained Code Generation**
   Reservoir simulation is a domain with *hard*, non-negotiable physical bounds. Violations are unambiguous and easy to diagnose.

Together, these make reservoir simulation code an ideal testbed for probing whether constraint knowledge is:

* explicitly represented,
* implicitly enforced,
* or absent at generation time.

---

## Unique Angle

* **Domain expertise** in Reservoir Engineering
* **Practical Eclipse simulation knowledge**
* **Mechanistic interpretability tooling (NNsight)**
* Focus on **token-level causal decisions**, not just outputs

This avoids purely surface-level evaluation and grounds the analysis in interpretable failure modes.

---

## Project Structure

```text
.
├── scripts/
│   ├── model_loader.py      # Loads Qwen model via NNsight
│   └── config.py            # HF cache and environment setup
├── prompts.py               # Capability check prompts and generation
├── analysis/
│   ├── token_analysis.py    # Token-level logit inspection
│   └── visualization.py     # Probability and token plots
├── README.md
```

---

## Model and Setup

* **Model:** Qwen/Qwen2.5-Coder-7B-Instruct
* **Decoding:** Greedy (do_sample=False, temperature=0)
* **Framework:** NNsight
* **Hardware:** Azure ML GPU (T4 / A100)
* **Prompt Format:** Native Qwen Instruct format

All experiments are run with deterministic decoding to ensure reproducibility.

---

## Phase 1: Capability Check (Baseline Validation)

Before studying failure modes, we verify that the model is **not “too dumb”** for the task (per Neel Nanda’s guidance).

### Capability Test Cases

| Case                | Description                  | Expected                             |
| ------------------- | ---------------------------- | ------------------------------------ |
| Valid Physics       | SWAT = 0.5                   | Correct Eclipse-style initialization |
| Valid + Constraints | SWAT = 0.5 + explicit bounds | Same as above                        |
| Invalid Physics     | SWAT = 1.5                   | Refusal or correction                |

### Key Result

* The model **successfully generates Eclipse-style initialization code** under valid physics.
* When prompted with **SWAT = 1.5**, the model **does not refuse or correct**, and instead generates a deterministic numeric continuation.

This establishes that:

* The model has **domain familiarity**
* The failure is **meaningful**, not due to ignorance

---

## Token-Level Finding (Key Insight)

Using NNsight, we isolate the exact generation step where the invalid value is produced.

* `"1.5"` is tokenized as `"1"`, `"."`, `"5"`
* The `"5"` token (ID 20) is selected with **near-unit probability**
* Competing tokens corresponding to refusal or constraint enforcement receive **negligible probability mass**

**Interpretation:**

> The violation arises from a confident numeric continuation, not from confusion or uncertainty. Physical constraints are not represented as competing hypotheses at the point of decision.

---

## Visualization Summary

The project includes:

* Token-probability bar charts at the violation step
* Bucketed probability mass (numeric continuation vs refusal vs other)
* Token position and ID tracing

These visuals are intentionally simple; “boring” plots here are strong evidence of deterministic failure.

---

## Time Tracking and Research Hygiene

All experiments are **time-boxed and tracked using Toggl**, following a phase-based research workflow.

* **Phase:** Capability Check
* **Duration:** ~1 hour
* **Tracking:** Single uninterrupted Toggl entry
* **Purpose:** Baseline validation before interpretability analysis

A Toggl screenshot is included in the application materials as evidence of disciplined research practice.

---

## Current Status

* ✅ Baseline capability verified
* ✅ Deterministic constraint violation identified
* ✅ Token-level causal locus established

---

## Next Phases (Planned)

1. **Representation Analysis**
   Does any layer encode physical feasibility signals?

2. **Intervention Experiments**
   Can suppressing numeric continuation logits induce refusal?

3. **Cross-Model Comparison**
   Do reasoning-specialized models behave differently?

---

## Takeaway

This project demonstrates that **constraint violations can arise from confident, locally optimal token decisions**, even when models appear to reason fluently. This raises important questions about the faithfulness of CoT explanations and the internal representation of hard constraints.

---
