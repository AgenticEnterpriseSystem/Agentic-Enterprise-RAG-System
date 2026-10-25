import json

import pytest

from scripts.benchmark_web import load_workloads, summarize


def test_custom_cases_strip_reference_fields_and_keep_original_questions(tmp_path):
    path = tmp_path/'cases.json'
    path.write_text(json.dumps([{'name': 'ticket', 'question': 'Why did jobs fail?',
                                 'gold_answer': 'Must never enter a request'}]))
    assert load_workloads(path) == (('ticket', 'Why did jobs fail?'),)
    assert summarize([], load_workloads(path))[0]['attempts'] == 0


@pytest.mark.parametrize('rows', [[{'name': '../escape', 'question': 'Q'}],
    [{'name': 'x', 'question': 'Q'}, {'name': 'x', 'question': 'Other Q'}], ['bad'], []])
def test_custom_case_names_cannot_overwrite_other_artifacts(tmp_path, rows):
    path = tmp_path/'cases.json'
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        load_workloads(path)
