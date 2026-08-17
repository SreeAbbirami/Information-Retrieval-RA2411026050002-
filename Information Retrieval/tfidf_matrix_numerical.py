import math
import sys
sys.stdout.reconfigure(encoding='utf-8')

def print_freq_matrix(documents, terms, frequency_matrix, title="Frequency Matrix"):
    print(f"\n{title}")
    print("       " + "  ".join([f"{t:>5}" for t in terms]))
    print("       " + "-" * (6 * len(terms)))
    for i, doc in enumerate(documents):
        row = "  ".join(f"{v:>5}" for v in frequency_matrix[i])
        print(f"  {doc}  {row}")

def compute_all(documents, terms, frequency_matrix):
    N = len(documents)
    total_terms_per_doc = [sum(row) for row in frequency_matrix]

    tf_matrix = []
    for i in range(N):
        row_tf = []
        for j in range(len(terms)):
            count = frequency_matrix[i][j]
            total = total_terms_per_doc[i]
            tf = count / total if total > 0 else 0.0
            row_tf.append(tf)
        tf_matrix.append(row_tf)

    doc_freq = [0] * len(terms)
    for j in range(len(terms)):
        df = 0
        for i in range(N):
            if frequency_matrix[i][j] > 0:
                df += 1
        doc_freq[j] = df

    idf_values = [0.0] * len(terms)
    for j in range(len(terms)):
        df = doc_freq[j]
        if df == 0:
            idf_values[j] = 0.0
        else:
            idf_values[j] = math.log(N / df)

    tfidf_matrix = []
    for i in range(N):
        row = []
        for j in range(len(terms)):
            row.append(tf_matrix[i][j] * idf_values[j])
        tfidf_matrix.append(row)

    return total_terms_per_doc, tf_matrix, doc_freq, idf_values, tfidf_matrix

def show_value_matrices(documents, terms, tf_matrix, idf_values, tfidf_matrix):
    def fmt_row(row):
        return "  ".join([f"{v:>7.4f}" for v in row])

    print(f"\n{'=' * 80}")
    print("STEP 1: Term Frequency (TF) Matrix")
    print(f"{'=' * 80}")
    print("Formula:  TF(t,d) = Count of t in d / Total terms in d")
    print("       " + "  ".join([f"{t:>7}" for t in terms]))
    print("       " + "-" * (8 * len(terms)))
    for i, doc in enumerate(documents):
        print(f"  {doc}  {fmt_row(tf_matrix[i])}")

    print(f"\n{'=' * 80}")
    print("STEP 2: Inverse Document Frequency (IDF)")
    print(f"{'=' * 80}")
    N = len(documents)
    print(f"Formula:  IDF(t) = ln( N / df(t) )   where N = {N}\n")
    for j, t in enumerate(terms):
        print(f"  IDF({t}) = ln({N}/{doc_freq[j]}) = {idf_values[j]:.6f}")

    print(f"\n{'=' * 80}")
    print("STEP 3: TF-IDF Matrix = TF x IDF")
    print(f"{'=' * 80}")
    print("       " + "  ".join([f"{t:>7}" for t in terms]))
    print("       " + "-" * (8 * len(terms)))
    for i, doc in enumerate(documents):
        print(f"  {doc}  {fmt_row(tfidf_matrix[i])}")

def calculate_one(documents, terms, frequency_matrix, total_terms_per_doc,
                  tf_matrix, doc_freq, idf_values, tfidf_matrix,
                  term_name, doc_name):

    if term_name not in terms:
        print(f"\n  [Error] Term '{term_name}' not found. Valid terms: {', '.join(terms)}")
        return None
    if doc_name not in documents:
        print(f"\n  [Error] Document '{doc_name}' not found. Valid docs: {', '.join(documents)}")
        return None

    j = terms.index(term_name)
    i = documents.index(doc_name)
    N = len(documents)

    count = frequency_matrix[i][j]
    total = total_terms_per_doc[i]
    tf = tf_matrix[i][j]
    df = doc_freq[j]
    idf = idf_values[j]
    tfidf = tfidf_matrix[i][j]

    print(f"\n{'=' * 80}")
    print(f"  CALCULATION: TF-IDF of Term {term_name} in Document {doc_name}")
    print(f"{'=' * 80}")
    print(f"  Given:")
    print(f"    -> Frequency count of {term_name} in {doc_name}  = {count}")
    print(f"    -> Total words in {doc_name}                    = {total}")
    print(f"    -> df({term_name}) = {df}  (appears in "
          f"{', '.join([documents[k] for k in range(N) if frequency_matrix[k][j] > 0]) or 'none'})")
    print(f"    -> Total documents N = {N}")
    print()
    print(f"  Step 1 - Term Frequency:")
    print(f"    TF({term_name}, {doc_name}) = count / total")
    print(f"                               = {count} / {total}")
    print(f"                               = {tf:.6f}")
    print()
    print(f"  Step 2 - Inverse Document Frequency:")
    print(f"    IDF({term_name}) = ln( N / df({term_name}) )")
    print(f"                     = ln( {N} / {df} )")
    if df > 0:
        print(f"                     = ln( {N/df:.4f} )")
    print(f"                     = {idf:.6f}")
    print()
    print(f"  Step 3 - TF-IDF:")
    print(f"    TF-IDF({term_name}, {doc_name}) = TF x IDF")
    print(f"                                   = {tf:.6f} x {idf:.6f}")
    print(f"                                   = {tfidf:.6f}")
    print()
    print(f"  [Final Answer]  TF-IDF({term_name} in {doc_name}) = {tfidf:.6f}")
    return tfidf


def input_int(prompt):
    while True:
        try:
            val = input(prompt).strip()
            if val == "":
                continue
            return int(val)
        except ValueError:
            print("  [!] Please enter a whole number.")

def input_row(prompt, expected_len):
    while True:
        raw = input(prompt).strip()
        parts = raw.replace(",", " ").split()
        try:
            nums = [int(x) for x in parts]
            if len(nums) == expected_len:
                return nums
            print(f"  [!] Expected {expected_len} numbers, got {len(nums)}. Try again.")
        except ValueError:
            print("  [!] Only integers allowed (separate by spaces or commas). Try again.")


def get_custom_matrix():
    print("\n--- Enter Your Custom Frequency Matrix ---")
    N_docs = input_int("Number of documents? (e.g. 5): ")
    N_terms = input_int("Number of terms? (e.g. 6): ")

    documents = [f"D{k+1}" for k in range(N_docs)]
    terms = [f"T{k+1}" for k in range(N_terms)]

    print(f"\nDocuments auto-named: {', '.join(documents)}")
    print(f"Terms auto-named:     {', '.join(terms)}")
    print("\nFor each document row, enter " + str(N_terms) + " integers (space or comma separated).")
    print("Example row for D1: 2 3 0 4 1 2\n")

    frequency_matrix = []
    for i, doc in enumerate(documents):
        row = input_row(f"  Row {doc}: ", N_terms)
        frequency_matrix.append(row)

    return documents, terms, frequency_matrix


DEFAULT_DOCS = ["D1", "D2", "D3", "D4", "D5"]
DEFAULT_TERMS = ["T1", "T2", "T3", "T4", "T5", "T6"]
DEFAULT_MATRIX = [
    [2, 3, 0, 4, 1, 2],
    [1, 0, 5, 0, 2, 0],
    [3, 1, 2, 2, 0, 4],
    [0, 2, 1, 3, 3, 4],
    [4, 0, 3, 1, 1, 0],
]

print()
print("=" * 80)
print("   INTERACTIVE TF-IDF CALCULATOR (Frequency Matrix Method)")
print("   Mahesh Huddar / VTU Style - Enter ANY matrix, calculate ANY Term-Document")
print("=" * 80)

print("""
  Choose an option:
    [1] Use the DEFAULT example matrix (5 docs, 6 terms)
    [2] Enter YOUR OWN custom frequency matrix
""")

documents = terms = frequency_matrix = None
while True:
    choice = input("  Enter 1 or 2: ").strip()
    if choice == "1":
        documents, terms, frequency_matrix = DEFAULT_DOCS, DEFAULT_TERMS, DEFAULT_MATRIX
        break
    elif choice == "2":
        documents, terms, frequency_matrix = get_custom_matrix()
        break
    else:
        print("  [!] Enter 1 or 2 only.")

print_freq_matrix(documents, terms, frequency_matrix)

total_terms_per_doc, tf_matrix, doc_freq, idf_values, tfidf_matrix = \
    compute_all(documents, terms, frequency_matrix)

print(f"\nTotal Terms per Document:")
for i, doc in enumerate(documents):
    parts = " + ".join(str(v) for v in frequency_matrix[i])
    print(f"  {doc}: {parts} = {total_terms_per_doc[i]}")

show_value_matrices(documents, terms, tf_matrix, idf_values, tfidf_matrix)

print(f"\n{'=' * 80}")
print("  NOW CALCULATE ANY RANDOM TERM + DOCUMENT COMBINATION")
print(f"{'=' * 80}")
print(f"  Documents: {', '.join(documents)}")
print(f"  Terms:     {', '.join(terms)}")
print()
print("  How to use:")
print("    -> At prompt, type any pair like:  T3 D2   or   T6 D4")
print("    -> Type 'ALL D2' to see full summary table for document D2")
print("    -> Type 'q' or 'quit' to exit")
print()

while True:
    try:
        line = input("Enter [Term] [Document] (or 'q' to quit): ").strip()
    except EOFError:
        print("\nGoodbye!\n")
        break

    if not line:
        continue
    if line.lower() in {"q", "quit", "exit"}:
        print("\nGoodbye!\n")
        break

    parts = line.replace(",", " ").split()
    if len(parts) != 2:
        print("  [!] Format: <Term> <Document>   Example: T3 D2")
        continue

    term_in, doc_in = parts[0].upper(), parts[1].upper()

    if term_in == "ALL":
        if doc_in not in documents:
            print(f"  [!] Document '{doc_in}' not found. Valid: {', '.join(documents)}")
            continue
        i = documents.index(doc_in)
        total = total_terms_per_doc[i]
        print(f"\n{'=' * 80}")
        print(f"  SUMMARY TABLE: All Terms in Document {doc_in}  (total words = {total})")
        print(f"{'=' * 80}")
        print(f"  {'Term':<6} {'Count':<8} {'TF':<12} {'df':<6} {'IDF':<12} {'TF-IDF':<12}")
        print("  " + "-" * 62)
        for j, t in enumerate(terms):
            print(f"  {t:<6} {frequency_matrix[i][j]:<8} "
                  f"{tf_matrix[i][j]:<12.6f} {doc_freq[j]:<6} "
                  f"{idf_values[j]:<12.6f} {tfidf_matrix[i][j]:<12.6f}")
        print()
        continue

    calculate_one(documents, terms, frequency_matrix, total_terms_per_doc,
                  tf_matrix, doc_freq, idf_values, tfidf_matrix,
                  term_in, doc_in)
