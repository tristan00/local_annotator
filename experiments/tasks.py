from local_annotator import Field

VERSION = "independent-ballots-v2"
TASKS = {
    "age": {"age": Field(
        "Extract only an explicitly stated CURRENT age of the comment author, in whole years. "
        "First-person age statements and self-identifying 25M, F/25, or 25-year-old count. "
        "Do not infer from dates, school, birth years, future birthdays, former ages, "
        "hypotheticals, quotations, or another person's age. Return not_found when no "
        "unambiguous current author age is explicit or current ages conflict.", "integer")},
    "appeal": {"appeal": Field(
        "Classify the predominant persuasive appeal of the author's own argument. "
        "logos: reasons, evidence, definitions, comparisons, or logical/causal inference; "
        "judge attempted reasoning, not truth. ethos: credibility, expertise, experience, "
        "trustworthiness, or character of the speaker or a cited source; 'I' alone is not ethos. "
        "pathos: emotions, suffering, fear, shame, sympathy, identity, or emotionally charged values; "
        "ethics alone is not pathos. Choose the appeal doing most persuasive work. "
        "mixed: multiple substantial appeals with none predominant. Return not_found for "
        "no discernible argument, neutral questions, greetings, bare reports, or fragments. "
        "Rhetorical questions can argue. Count quotations only when the author endorses or "
        "uses them to persuade. Do not invent parent context.",
        allowed=("ethos", "pathos", "logos", "mixed"))},
}
