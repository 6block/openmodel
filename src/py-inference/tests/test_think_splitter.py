"""_ThinkSplitter: the thinking-mode stream splitter.

The splitter turns "<think>REASONING</think>ANSWER" into separate
reasoning/content channels while tolerating tags split across stream deltas.
It is instantiated only when a client opts in, so these tests also pin the
worst-case chunkings (character-by-character) that real streams produce.
"""
from inference.api_server import _ThinkSplitter


def run(deltas):
    sp = _ThinkSplitter()
    reasoning, content = "", ""
    for d in deltas:
        r, c = sp.feed(d)
        reasoning += r
        content += c
    r, c = sp.flush()
    return reasoning + r, content + c


def test_whole_block_single_delta():
    assert run(["<think>\nreason\n</think>\n\nanswer"]) == ("reason", "answer")


def test_tags_split_across_deltas():
    assert run(["<th", "ink>\nab", "c</th", "ink>\n\nok"]) == ("abc", "ok")


def test_plain_answer_passthrough():
    assert run(["plain ", "answer"]) == ("", "plain answer")


def test_truncated_inside_reasoning():
    assert run(["<think>\nonly reasoning so far"]) == ("only reasoning so far", "")


def test_leading_whitespace_before_tag():
    assert run(["\n<think>x</think>done"]) == ("x", "done")


def test_character_by_character_stream():
    assert run(list("<think>\nab\n</think>\nZ")) == ("ab", "Z")


def test_interior_newlines_preserved():
    assert run(["<think>\nline1\n\nline2\n</think>\nanswer"]) == ("line1\n\nline2", "answer")


def test_truncated_mid_closing_tag_drops_partial_markup():
    assert run(["<think>\nabc</thi"]) == ("abc", "")


def test_close_tag_lookalike_in_answer_untouched():
    assert run(["<think>a</think>see </think> literal"]) == ("a", "see </think> literal")


def run_pre_opened(deltas):
    sp = _ThinkSplitter(pre_opened=True)
    reasoning, content = "", ""
    for d in deltas:
        r, c = sp.feed(d)
        reasoning += r
        content += c
    r, c = sp.flush()
    return reasoning + r, content + c


def test_pre_opened_template_generation_starts_in_reasoning():
    # Newer Qwen templates pre-fill "<think>\n" in the prompt: the generation
    # begins straight in the reasoning and only carries the CLOSING tag.
    assert run_pre_opened(['The user said "hi". Brief reply.\n</think>\n\nHey!']) == (
        'The user said "hi". Brief reply.', "Hey!")


def test_pre_opened_character_stream():
    assert run_pre_opened(list("ab\n</think>\nZ")) == ("ab", "Z")


def test_pre_opened_truncated_reasoning():
    assert run_pre_opened(["half a thought"]) == ("half a thought", "")


def run_harmony(deltas):
    sp = _ThinkSplitter(open_tag="analysis", close_tag="assistantfinal")
    reasoning, content = "", ""
    for d in deltas:
        r, c = sp.feed(d)
        reasoning += r
        content += c
    r, c = sp.flush()
    return reasoning + r, content + c


def test_harmony_real_sample():
    # Verbatim gpt-oss output observed in production (special tokens stripped).
    real = 'analysisUser asks: "What is 6*7? Brief." They want answer: 42. Provide brief answer.assistantfinal42'
    assert run_harmony([real]) == (
        'User asks: "What is 6*7? Brief." They want answer: 42. Provide brief answer.', "42")


def test_harmony_character_stream():
    assert run_harmony(list("analysisab.assistantfinalOK")) == ("ab.", "OK")


def test_harmony_truncated_in_analysis():
    assert run_harmony(["analysishalf a thought"]) == ("half a thought", "")


# --- _split_completed: what actually reaches the caller ---------------------
#
# The splitter above is only half the story: a default (non-thinking) request
# discards the reasoning channel, so a generation truncated before it produced
# any answer used to return content="" AND reasoning_content=null — nothing at
# all, for tokens the caller was billed for. These pin the rescue and, just as
# importantly, that it does not leak reasoning on ordinary requests.

from inference.api_server import _split_completed


def harmony_splitter():
    return _ThinkSplitter(open_tag="analysis", close_tag="assistantfinal")


def test_split_completed_no_splitter_passes_text_through():
    assert _split_completed("plain answer", None, False) == (None, "plain answer")


def test_split_completed_harmony_default_request_hides_reasoning():
    text = 'analysisThey want 6*7 = 42.assistantfinal42'
    assert _split_completed(text, harmony_splitter(), False) == (None, "42")


def test_split_completed_harmony_thinking_request_surfaces_reasoning():
    text = 'analysisThey want 6*7 = 42.assistantfinal42'
    assert _split_completed(text, harmony_splitter(), True) == ("They want 6*7 = 42.", "42")


def test_split_completed_truncated_in_reasoning_is_rescued():
    # max_tokens ran out inside the analysis channel: no answer exists. The
    # caller asked for a plain answer, but returning nothing is worse than
    # returning the reasoning they paid for.
    text = "analysisLet me compare 9.11 and 9.9 digit by digit"
    reasoning, content = _split_completed(text, harmony_splitter(), False)
    assert content == ""
    assert reasoning == "Let me compare 9.11 and 9.9 digit by digit"


def test_split_completed_rescue_does_not_fire_when_an_answer_exists():
    # Truncated in the ANSWER, not the reasoning: a partial answer is present,
    # so the reasoning stays hidden on a default request.
    text = 'analysisthinking hereassistantfinalThe answer is 9.9 because'
    assert _split_completed(text, harmony_splitter(), False) == (
        None, "The answer is 9.9 because")


def test_split_completed_qwen_thinking_truncated_keeps_reasoning():
    # Qwen dialect, thinking explicitly on: unchanged behaviour, reasoning is
    # surfaced because it was asked for.
    reasoning, content = _split_completed(
        "<think>\nhalf a thought", _ThinkSplitter(), True)
    assert (reasoning, content) == ("half a thought", "")


def test_split_completed_empty_generation():
    assert _split_completed("", harmony_splitter(), False) == (None, "")
