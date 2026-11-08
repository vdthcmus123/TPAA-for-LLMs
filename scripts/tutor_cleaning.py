"""Shared cleaning + QA for generated tutor responses (one tutor turn).

Used verbatim inside notebooks/molab_tpaa_generation.py (cell "Làm sạch & QA")
and scripts/run_api_tutors.py. Cleaning only fixes FORMAT; wording is never changed.
"""
import re

CLEANING_VERSION = "tutor_clean_v1"

# A new dialogue turn invented by the model: "Student:", "Student response:",
# "Tutor (encouraging):", "Teacher response:" ... not at the very start.
TURN_TAG_RE = re.compile(
    r"(?:^|(?<=[\s\"'*(]))(?:\*\*)?(Student|Tutor|Teacher)(?:'s)?(?:\s+(?:response|reply))?"
    r"(?:\s*\([^)]{0,40}\))?(?:\*\*)?\s*:",
    re.IGNORECASE,
)
LEADING_TAG_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:(?:Tutor|Teacher)(?:'s)?(?:\s+(?:response|reply))?|Response|Answer)"
    r"(?:\s*\([^)]{0,80}\))?\s*(?:\*\*)?\s*:\s*(?:\*\*)?\s*",
    re.IGNORECASE,
)
REASONING_RE = re.compile(
    r"^\s*(reasoning|analysis|thought process)\b|^\s*step \d+\s*:|\n\s*(reasoning|analysis)\s*:",
    re.IGNORECASE,
)
SPECIAL_RE = re.compile(r"<\|[^>]*\|>|</?think>|\[/?INST\]|</?s>")
META_RE = re.compile(
    r"\b(the tutor (should|needs|must|can|will)|the (tutor'?s? )?response should|"
    r"the student (is|has|was|needs|made|seems|confus)|student'?s mistake|pedagogical|"
    r"as (a|the) tutor,? I)\b",
    re.IGNORECASE,
)
MARKDOWN_RE = re.compile(r"\*\*|__|`|^\s*#+\s*|^\s*[-*•]\s+", re.MULTILINE)
# *word* emphasis (not multiplication like "6 * 10": the * must touch the word on both sides)
ITALIC_RE = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")


PREAMBLE_RE = re.compile(
    r"^\s*(?:the )?student\b|^\s*the correct answer is\b|"
    r"\b(?:a|the) (?:tutor'?s? )?response should\b|\bthe tutor should\b",
    re.IGNORECASE,
)


def drop_preamble(text):
    """Model wrote an analysis paragraph first and the real tutor turn after it
    (labelled "Tutor:" or in quotes): keep the tutor turn, drop the analysis."""
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(paras) < 2 or not (PREAMBLE_RE.search(paras[0]) or META_RE.search(paras[0])):
        return text, False
    nxt = paras[1].strip()
    if LEADING_TAG_RE.match(nxt) or nxt[:1] in "\"“":
        return "\n\n".join(paras[1:]), True
    return text, False


def find_new_turn(text):
    """Index where an invented next turn starts (after some real content), or -1."""
    for m in TURN_TAG_RE.finditer(text):
        if text[: m.start()].strip(" \t\n*\"'"):
            return m.start()
    return -1


def clean_response(raw, hit_max_tokens=False):
    """Return (clean_text, actions). Only formatting is changed."""
    actions = []
    text = str(raw or "")
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
        actions.append("dropped_reasoning_block")
    stripped = SPECIAL_RE.sub("", text)
    if stripped != text:
        actions.append("removed_special_tokens")
        text = stripped
    text = text.strip()
    text, _dropped = drop_preamble(text)
    if _dropped:
        actions.append("dropped_preamble")
    new = LEADING_TAG_RE.sub("", text, count=1)
    if new != text:
        actions.append("stripped_leading_role_tag")
        text = new
    cut = find_new_turn(text)
    if cut >= 0:
        text = text[:cut]
        actions.append("cut_invented_next_turn")
    parts = re.split(r"\n\s*\n", text.strip(), maxsplit=1)
    if len(parts) > 1 and parts[1].strip():
        text = parts[0]
        actions.append("cut_after_first_paragraph")
    new = ITALIC_RE.sub(r"\1", MARKDOWN_RE.sub("", text)).replace("\\*", "*")
    if new != text:
        actions.append("removed_markdown")
        text = new
    if "\n" in text.strip():
        actions.append("joined_lines")
    text = re.sub(r"\s+", " ", text).strip()
    q = text.strip("\"“”'")
    if q != text and text[:1] in "\"“'" and text[-1:] in "\"”'":
        text = q.strip()
        actions.append("removed_wrapping_quotes")
    if hit_max_tokens and text and text[-1] not in ".?!\"”)":
        last = max(text.rfind(". "), text.rfind("? "), text.rfind("! "))
        if last > 0:
            text = text[: last + 1]
            actions.append("trimmed_incomplete_sentence")
    return text, actions


def qa_check(text, method="", thinking_tokens=None, reasoning_closed=None, hit_max_tokens=False):
    """Return (errors, info). errors must be empty for a usable row."""
    errors, info = [], []
    t = str(text or "")
    words = len(t.split())
    if not t.strip():
        errors.append("empty")
    if SPECIAL_RE.search(t):
        errors.append("special_tokens_left")
    if LEADING_TAG_RE.match(t) or find_new_turn(t) >= 0:
        errors.append("role_tag_left")
    if re.match(r"^\s*(?:\*\*)?\s*Student(?:'s)?\b[^:\n]{0,30}:", t, re.IGNORECASE):
        errors.append("response_is_student_turn")
    if "**" in t or "\n" in t:
        errors.append("formatting_left")
    if REASONING_RE.search(t) or (method == "cot" and META_RE.search(t)):
        errors.append("reasoning_leaked")
    elif META_RE.search(t):
        info.append("third_person_meta_comment")
    if t and t[-1] not in ".?!\"”)'":
        (errors if hit_max_tokens else info).append("no_sentence_end")
    number_run = re.search(r"(?:\b\d+\s*,\s*){12,}", t)
    long_number_run = re.search(r"(?:\b\d+\s*,\s*){30,}", t)
    if long_number_run or (number_run and hit_max_tokens) or re.search(
        r"(\b\w+\b(?:\W+\w+){0,3})(?:\W+\1\b){5,}", t, re.I
    ):
        errors.append("degenerate_repetition")
    elif number_run:
        info.append("long_number_list")
    if method == "cot":
        if not reasoning_closed:
            errors.append("cot_reasoning_not_closed")
        elif (thinking_tokens or 0) < 10:
            errors.append("cot_no_reasoning")
    if words > 80:
        info.append("long_over_80_words")
    if len(re.findall(r"[.!?](?:\s|$)", t)) > 1:
        info.append("more_than_one_sentence")
    return errors, info
