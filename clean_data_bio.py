import re

input_path = "biophysics10_BIO.txt"
output_path = "c_biophysics10_BIO.txt"

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

cleaned_lines = []


prev_tag = "O"
term_buffer = []
max_term_len = 4


common_non_terms = {
    "page", "method", "model", "new", "results", "structure", "is", "are", "was", "were",
    "it", "this", "that", "these", "those", "we", "they", "you", "i", "can", "may",
    "the", "and", "or", "of", "to", "in", "on", "as", "with", "by", "for", "from", "at",
    "TABLE", "Fig", "APPENDIX", "REFERENCES"
}

for i, line in enumerate(lines):
    line = line.strip()

    if not line:
        if len(term_buffer) > max_term_len:
            for j in range(max_term_len, len(term_buffer)):
                term_buffer[j] = term_buffer[j].split()[0] + " O"
        cleaned_lines.extend(term_buffer)
        cleaned_lines.append("")
        prev_tag = "O"
        term_buffer = []
        continue

    parts = line.split()
    if len(parts) != 2:
        continue

    word, tag = parts

    if re.search(r"(http|\.com|github\.io)", word.lower()):
        continue
    if re.match(r"\w+@\w+\.\w+", word):
        continue
    if re.match(r"^[A-Z][a-z]+[0-9,]*$", word) and tag.startswith("B-"):
        continue
    if word in {"Inria", "CNRS", "France", "Univ", "ENPC", "LIGM"}:
        continue

    # 删除无意义字符或格式化符号
    if re.match(r"^\d{4,}$", word):
        continue
    if re.match(r"^[.,;:()\[\]{}]$", word) and tag != "O":
        continue
    if len(word) == 1 and tag != "O":
        continue

    w_lower = word.lower()

    if w_lower in common_non_terms and tag != "O":
        tag = "O"

    if tag == "I-TERM" and prev_tag == "O":
        tag = "O"

    if tag.startswith("B-"):
        term_buffer = [f"{word} {tag}"]
        prev_tag = tag
    elif tag.startswith("I-"):
        term_buffer.append(f"{word} {tag}")
        prev_tag = tag
    else:
        if term_buffer:
            if len(term_buffer) > max_term_len:
                for j in range(max_term_len, len(term_buffer)):
                    term_buffer[j] = term_buffer[j].split()[0] + " O"
            cleaned_lines.extend(term_buffer)
            term_buffer = []
        cleaned_lines.append(f"{word} {tag}")
        prev_tag = tag

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(cleaned_lines))

