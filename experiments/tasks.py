from local_annotator import Field

VERSION = "reddit-v1-20261025"
TASKS = {
    "age": {"age": Field(
        "Extract the author's explicitly stated CURRENT age in whole years, as a JSON integer. "
        "First-person 'I am 25', '25M', 'F/25', or 'as a 25-year-old' count. "
        "Do not infer age from birth year, school stage, dates, another person's age, quoted speakers, "
        "hypotheticals, former ages, or future birthdays. If multiple current ages conflict, "
        "or no unambiguous explicit current author age is stated, return not_found. "
        "Use only this comment, without parent context or outside knowledge.", "integer")},
    "appeal": {"appeal": Field(
        "Label the predominant rhetorical appeal of THIS comment's own argument, using only this text. "
        "logos: persuasion primarily through reasons, evidence, definitions, comparisons, causal or "
        "logical inference; assess the attempted reasoning, not whether its claims are true. "
        "ethos: persuasion primarily through the speaker's or a cited source's credibility, expertise, "
        "experience, trustworthiness or character. Merely using 'I' does not make ethos. "
        "pathos: persuasion primarily through emotions, vivid suffering, fear, shame, sympathy, "
        "identity or emotionally charged values. Merely discussing ethics does not make pathos. "
        "Choose the appeal doing most of the persuasive work, even when another is present. "
        "Use mixed only when two or more appeals are substantial and no single appeal predominates. "
        "Use not_found for no discernible persuasive argument (e.g. a neutral question, greeting, "
        "bare factual report, or contextless fragment). A rhetorical question can make an argument. "
        "Judge quoted material only insofar as the author endorses or deploys it; do not attribute "
        "opponents' quoted appeals to the author. Do not infer missing parent context.",
        allowed=("ethos", "pathos", "logos", "mixed"))},
}
