# Five-Class Hinglish Sentiment Intensity Annotation Guidelines (v1.0)

**Project Title:** Hinglish Sentiment Classification: Investigating Minority-Class Collapse and Translation Effects  
**Document Status:** Frozen Rubric v1.0 (Source of Truth)

---

## 1. Class Definitions and ID Mapping

Every row is mapped to exactly one of the five ordinal classes below:

| ID | Code Name | Label Name | Definition | Diagnostic Cues (Supporting Only) |
|---|---|---|---|---|
| **0** | `strongly_negative` | Strongly Negative | Intense dissatisfaction, anger, harsh criticism, accusation, outrage, or emphatic negative verdict. | Absolute wording, explicit anger, intent never to repeat or to warn others ("bilkul bakwaas", "fraud", "loot liya", "barbaad", "hated"). |
| **1** | `weakly_negative` | Weakly Negative | Mild dissatisfaction, slight disappointment, criticism, complaint, or slightly negative opinion. | Hedges, diminishers, soft criticism, polite dissatisfaction ("thoda slow hai", "kuch khas nahi", "boring lag raha tha", "could be better"). |
| **2** | `neutral` | Neutral | Factual, informational, instructional, technical, or non-evaluative statement; query without sentiment. | Definitions, specifications, meeting announcements, scripture quotes without personal stance, instructions. |
| **3** | `weakly_positive` | Weakly Positive | Mild approval, hedged praise, moderate satisfaction, or a slightly positive opinion. | Diminishers, qualified praise, mild satisfaction ("theek-thaak accha hai", "decent", "chalega", "bura nahi hai", "ek baar try kar sakte"). |
| **4** | `strongly_positive` | Strongly Positive | Strong praise, excitement, enthusiasm, gratitude, intense appreciation, or emphatic positive verdict. | Superlatives, explicit delight, strong recommendation, profound gratitude ("zabardast", "paisa vasool", "shandar", "masterpiece", "bahut bahut dhanyavad"). |

---

## 2. Decision Procedure (Applied in Strict Order)

1. **Step 1 — Target & Stance:** Identify what is evaluated and whether the writer has an evaluative stance. If strictly factual, technical, or informational $\rightarrow$ assign **Neutral (2)**.
2. **Step 2 — Polarity:** If an evaluative stance exists, decide between Negative or Positive polarity, properly accounting for negation scope, sarcasm, and code-mixed discourse.
3. **Step 3 — Intensity:** Decide between **Strong** versus **Weak** based on the overall semantic force of the evaluation:
   - *Strong:* Extreme wording, unreserved verdicts, emotional superlatives, uncompromising stance.
   - *Weak:* Hedges (*thoda*, *kuch had tak*), conditional praise/criticism, understated tone, concessions.
4. **Step 4 — Flags Assignment:** Record all applicable metadata flags:
   - `sarcasm`: Sarcastic or ironic tone where literal meaning differs from intended meaning.
   - `negation`: Explicit negation affecting sentiment ("nahi", "mat", "not", "kabhi nahi").
   - `mixed`: Contains both positive and negative aspects; label the predominant stance.
   - `slang`: Informal colloquial Hinglish slang (*bakwaas*, *jugaad*, *mast*, *bindaas*).
   - `code_mix`: Frequent intrasentential switching between Hindi and English words.
   - `short_context`: Fragmented or very brief sentence where context is limited.
   - `uncertain`: Boundary borderline case where intensity cannot be determined unambiguously.
   - `orig_conflict`: Final annotated polarity contradicts the original 3-class label.

---

## 3. Rules for Difficult Text

| Case | Annotation Rule | Correct Interpretation |
|---|---|---|
| **Keywords $\neq$ Sentiment** | Do not label purely by single keywords. Words like *accha* can be conversational acknowledgements. | "Accha ji, samajh gaya." $\rightarrow$ **Neutral (2)** |
| **Negation Scope** | Resolve negation scope before polarity. Negated negatives are usually Weakly Positive; negated positives are Weakly Negative. | "Bura nahi hai." $\rightarrow$ **Weakly Positive (3)**<br>"Acche din nahi aaye." $\rightarrow$ **Weakly Negative (1)** |
| **Mixed Sentiment** | Determine writer's overall verdict on primary target. If genuinely balanced, mark Neutral with `mixed` flag. | "Camera achha hai lekin battery bilkul bekar hai." $\rightarrow$ **Weakly Negative (1)** (`mixed`) |
| **Sarcasm / Irony** | Annotate intended stance, not literal words. Flag with `sarcasm`. If undecidable without outside context, flag `uncertain`. | "Wah, kya shandaar service hai, sirf teen ghante intezaar karwaya." $\rightarrow$ **Strongly Negative (0)** |
| **Spelling Variants** | Treat orthographic variants as identical lexemes. | *masst*, *mast*, *masttt* $\rightarrow$ same underlying root. |
| **Tie-Breaker Rule** | Never default silently. Mark `uncertain` flag and let secondary adjudicator resolve. | Ensure transparent documentation of subjective boundaries. |
