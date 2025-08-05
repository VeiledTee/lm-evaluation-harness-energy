import datasets
import re
import string
import collections

def squad_normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"(a|an|the)", " ", text)
    tokens = text.split()
    return " ".join(tokens)

def squad_exact(a: str, b: str) -> int:
    return int(squad_normalize(a) == squad_normalize(b))

def squad_f1(a: str, b: str) -> float:
    a_tokens = squad_normalize(a).split()
    b_tokens = squad_normalize(b).split()
    common = collections.Counter(a_tokens) & collections.Counter(b_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(a_tokens)
    rec = num_same / len(b_tokens)
    return 2 * prec * rec / (prec + rec)

def _max_metric_over_gold_list(golds, pred, fn):
    if isinstance(golds, list):
        return max((fn(g, pred) for g in golds), default=0.0)
    return fn(golds, pred)


def em_squad_max_impl(items):
    """
    SQuAD-style EM: normalize and compare; if multiple golds, take max.
    """
    gold, pred = items[0], items[1]
    return float(_max_metric_over_gold_list(gold or [], pred or "", squad_exact))

def f1_squad_max_impl(items):
    """
    SQuAD-style F1: token-level F1 with normalization; max over golds.
    """
    gold, pred = items[0], items[1]
    return float(_max_metric_over_gold_list(gold or [], pred or "", squad_f1))


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process(doc):
        contexts = []
        for title, sentences in zip(
            doc.get('context', {}).get('title', []),
            doc.get('context', {}).get('sentences', [])
        ):
            contexts.append(f"{title}: {' '.join(sentences)}")
        answer = doc.get('answer')
        if answer is None:
            answer = ""
        return {
            'question': doc.get('question', ''),
            'context': contexts,
            'answer': answer
        }
    return dataset.map(_process, remove_columns=dataset.column_names)


def doc_to_text_with_context(doc: dict) -> str:
    # Build prompt with question and supporting contexts
    ctx = "".join(f"- {c}" for c in doc.get('context', []))
    return f"Question: {doc.get('question', '')}\nContext: {ctx}\nAnswer:"


def doc_to_text_without_context(doc: dict) -> str:
    # Build a prompt with only the question
    return f"Question: {doc.get('question', '')}\nAnswer:"


def doc_to_target(doc: dict) -> str:
    ans = doc.get('answer', '')
    if isinstance(ans, list) and ans:
        ans = ans[0]
    if ans is None:
        ans = ''
    return " ".join(str(ans).split())