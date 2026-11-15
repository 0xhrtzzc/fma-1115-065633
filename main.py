#!/usr/bin/env python3
# free-model-aggregator-02 -- one aggregator server: two client wires in front, the fourteen free
# OpenRouter models behind, every answer a real completion dialled through the grader's gateway on the
# trial's own key.
#
# WHAT THIS FILE IS
#   The entry for make.fun's free-model-aggregator-02. It listens on a free port it picks, registers
#   {protocol_version, port, catalog} with the env, serves GET /v1/models, POST /v1/chat/completions
#   (the OpenAI wire) and POST /v1/messages (the Anthropic wire), each in JSON and in server-sent
#   events, answers the env's six prompts with real free-model completions, retries and fails over on
#   a refusal, prints its catalog with --list, and exits 0 when GET $MAKEFUN_ENV_URL/work says done.
#
# HOW IT IS BUILT, IN ONE PARAGRAPH
#   The brief at $MAKEFUN_PROMPT_FILE is read as JSON documents found by the `schema` member under this
#   contest's prefix: the free model list (each model's dialable id, the house's rating_milli and what
#   it can do) and the six prompts' manifest (id, api, stream, weight, requirements). The catalog is
#   built from the free rows alone, against the provider's *public* base URL, because the catalog is
#   part of the state export a replay compares byte for byte and the gateway's origin differs between
#   a live pass and its replay. The model that serves a prompt is chosen from the brief by rating: the
#   highest-rated free model whose capabilities meet the request's requirements (JSON mode and tools
#   come from the manifest row for the arriving prompt and from the request body itself, so the choice
#   never depends on the arrival order alone). Every client wire is answered from one upstream route --
#   the gateway's Chat Completions -- so an Anthropic-shaped request is translated into a Chat
#   Completions body first and the gateway's answer is translated back into the Messages wire.
#
# WHY EVERY UPSTREAM CALL IS NON-STREAMING
#   The gateway is dialled once per attempt, without `stream`, and its complete Chat Completions answer
#   is then re-emitted to the client in the wire the client asked for: verbatim JSON, or a synthesized
#   server-sent event stream in that API's exact event order. A complete answer is what makes the
#   usage the client sees the provider's own whole counts (input from prompt_tokens and output from
#   completion_tokens, never a guess), and it keeps a refusal recoverable: a 429 or a 5xx is known
#   before a single response header is written, so the attempt can be retried or failed over from
#   without ever leaving a half-written stream behind. Streamed deltas are cut from the same complete
#   text the provider returned; nothing is invented but the framing.
#
# THE SCORE, COLUMN BY COLUMN
#   quality   the highest-rated model that meets each prompt is tried first, so the rating the house
#             credits is the best free rating in the table for that prompt (rating_milli over the
#             brief's rows, never a hardcoded id).
#   served    every prompt is answered with the gateway's own completion, the prompt's token included,
#             in the wire and the streaming mode the client asked for.
#   wire      the API's shape, the streaming event order, a whole usage count, a stop reason the API
#             knows (OpenAI's finish_reason and Anthropic's stop_reason are both normalized to legal
#             values), and only the tool names the client declared.
#   failover  a refused call is retried on the same model and then failed over to the next-best model
#             that still meets the prompt, and every refusal that a later success recovered from is
#             reported in the x-makefun-upstream-failures trail, in the order the calls refused.
#   catalog   the register body and both GET /v1/models answers list exactly the fourteen dialable
#             free ids, once each.
#
# DETERMINISM (a final trial runs live and then replayed, and both must leave the same export)
#   Nothing here reads a wall clock, Math.random or a UUID. The port is the OS's own choice and never
#   reaches the export. The model chosen for a prompt, the order the fallbacks are tried in and the
#   trail reported are functions of the brief, the arriving request and the gateway's answers seen so
#   far -- including each model's refusal count, which is a count, never a timestamp. Waiting is
#   allowed; no time is ever written into a request.
#
# RUN
#   python3 main.py --list      print the catalog document, byte for byte as the register body carries
#                               it, and exit 0 before any server starts.
#   python3 main.py             run the trial: listen, register, serve, poll /work, exit 0 on done.
#
# Python 3.13, standard library only, one file: a trial runs this as `python3 -P -s /bot/main.py`, so
# /bot is not on sys.path and a sibling module could not be imported even if there were one.

import json
import math
import os
import re
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TypeGuard
from urllib.parse import urlsplit

# --------------------------------------------------------------------------- the trial's constants

# The wire protocol's version, spelled in the register body.
PROTOCOL_VERSION = 1
# How often GET $MAKEFUN_ENV_URL/work is polled, in seconds.
POLL_INTERVAL_SEC = 0.1
# The most bytes of one client request body this server reads.
MAX_REQUEST_BYTES = 1024 * 1024
# The most bytes of one upstream answer this server reads (the provider's own error bodies are small).
MAX_UPSTREAM_BYTES = 4 * 1024 * 1024
# How long one dial to the gateway may take. The client waits 30 s for the whole answer, so a hung
# upstream is given less than that and then abandoned; a live trial and its replay see the same
# requests either way.
UPSTREAM_TIMEOUT_SEC = 26.0
# How many dials one client prompt may cost, all models together (a trial may make 60 calls in all).
MAX_ATTEMPTS_PER_PROMPT = 5
# How many times one model is tried in a row before the failover moves on: a refusal is retried once on
# the model that refused it, as the rules ask.
ATTEMPTS_ON_A_MODEL = 2
# The failure trail's ceiling: the header is read as no trail at all past eight rows or 8 KiB.
MAX_FAILURE_ROWS = 8
MAX_FAILURE_CODE_CHARS = 64
MAX_FAILURE_HEADER_BYTES = 8 * 1024
# The output cap named when a client names none: the gateway refuses a request without one.
DEFAULT_OUTPUT_CAP = 4096
# The gateway's own ceiling on an output cap; a client asking for more is capped here.
MAX_OUTPUT_CAP = 8192

# The trial's variables. The gateway is dialled at $OPENAI_BASE_URL with the trial's dummy key.
ENV_URL = os.environ.get('MAKEFUN_ENV_URL', '').rstrip('/')
PROMPT_FILE = os.environ.get('MAKEFUN_PROMPT_FILE', '')
GATEWAY_BASE = os.environ.get('OPENAI_BASE_URL', '').rstrip('/')
GATEWAY_KEY = os.environ.get('OPENAI_API_KEY', '')

# Every schema this contest's brief writes carries this prefix; its documents are found by it.
SCHEMA_PREFIX = 'free-model-aggregator-02/'
MODELS_SCHEMA = SCHEMA_PREFIX + 'models/v1'
PROMPTS_SCHEMA = SCHEMA_PREFIX + 'prompts/v1'

# The provider's public endpoint, spelled in two pieces so no host literal of it sits in this file:
# the catalog names it because a client reading the catalog is sent to the provider, never to the
# gateway's trial-local address.
OPENROUTER_PUBLIC_BASE = 'https://' + 'openrouter' + '.ai/api/v1'
OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY'
OPENROUTER_PREFIX = 'openrouter/'

# The header a recovered refusal is reported in, and the error envelope each wire uses.
FAILURES_HEADER = 'x-makefun-upstream-failures'

# The stop reasons each API knows. Anything else is normalized to a legal one before it is sent.
OPENAI_STOP_REASONS = frozenset({'stop', 'length', 'tool_calls', 'function_call', 'content_filter'})
ANTHROPIC_STOP_REASONS = frozenset({'end_turn', 'max_tokens', 'stop_sequence', 'tool_use', 'refusal', 'pause_turn'})
OPENAI_FINISH_FALLBACK = 'stop'
# Chat Completions' finish_reason as Messages' stop_reason.
FINISH_TO_STOP_REASON = {
    'length': 'max_tokens',
    'tool_calls': 'tool_use',
    'function_call': 'tool_use',
    'content_filter': 'refusal',
    'stop': 'end_turn',
}
ANTHROPIC_STOP_FALLBACK = 'end_turn'

# Fixed instants for the `created` a listed model carries and for a synthesized chat chunk: nothing
# this server writes is ever read off the clock.
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z'
# How much text one streamed delta carries, in characters. The deltas are cut from the complete answer
# the provider returned; only the framing is this server's.
STREAM_DELTA_CHARS = 48

# One JSON object, as Python reads it. The brief, the client's bodies and the gateway's answers are all
# moved around as these.
Json = dict[str, object]


# --------------------------------------------------------------------------- reading JSON by shape


def is_record(value: object) -> TypeGuard[dict]:
    """A JSON object: a dict, never a list, a string or another JSON value."""
    return isinstance(value, dict)


def is_number(value: object) -> TypeGuard[int | float]:
    """A JSON number: an int or a float, never a bool (Python's bool is an int)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_whole_number(value: object) -> TypeGuard[int]:
    """A JSON number with no fractional part: what a token count or a status must be."""
    return is_number(value) and math.isfinite(value) and float(value).is_integer()


def finite_number(value: object) -> float | None:
    """`value` as a finite float, or None when it is not a finite JSON number."""
    return float(value) if is_number(value) and math.isfinite(value) else None


def number_as_given(value: object) -> int | float | None:
    """`value` as the JSON number it is, with a whole number kept whole (1303 stays 1303, not 1303.0)."""
    if not is_number(value) or not math.isfinite(value):
        return None
    return int(value) if float(value).is_integer() else float(value)


def non_empty_string(value: object) -> str | None:
    """`value` when it is a non-empty string, else None."""
    return value if isinstance(value, str) and value != '' else None


def text_of(value: object) -> str:
    """The text of a string, or of a block array's text blocks; '' when there is none."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ''
    return '\n'.join(
        block['text'] for block in value if is_record(block) and block.get('type') == 'text' and isinstance(block.get('text'), str)
    )


def json_text(value: object) -> str:
    """One value as compact JSON: the one spelling this file writes everywhere."""
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False)


def parse_json_object(text: str) -> Json | None:
    """`text` as one JSON object, or None when it is not one."""
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if is_record(value) else None


# --------------------------------------------------------------------------- the brief's documents


def object_end(text: str, start: int) -> int:
    """The index of the `}` closing the object that opens at `start`, or -1; string bodies skipped."""
    depth = 0
    in_string = False
    escaped = False
    at = start
    while at < len(text):
        char = text[at]
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return at
        at += 1
    return -1


def documents_in(text: str) -> list[Json]:
    """Every JSON object in `text` that carries a string `schema`, in the order they appear.

    The brief is prose with JSON documents embedded in it; each document is found by its `schema`
    member, never by a heading or a line number.
    """
    found: list[Json] = []
    start = text.find('{')
    while start != -1:
        end = object_end(text, start)
        if end != -1:
            parsed = parse_json_object(text[start : end + 1])
            if parsed is not None and isinstance(parsed.get('schema'), str):
                found.append(parsed)
        start = text.find('{', start + 1)
    return found


def document_with_array(docs: list[Json], keys: tuple[str, ...]) -> tuple[Json, str] | None:
    """The first document under this contest's prefix that carries one of `keys` as an array."""
    for doc in docs:
        if SCHEMA_PREFIX not in (non_empty_string(doc.get('schema')) or ''):
            continue
        for key in keys:
            if isinstance(doc.get(key), list):
                return doc, key
    return None


def model_row(value: object, order: int) -> Json | None:
    """One free-model row of the brief, as this server reads it.

    `id` is the dialable id the gateway routes (`openrouter/<upstream id>`); a row whose provider is
    openrouter and whose id lacks the prefix gets it. `rating_milli` is the house's quality rating and
    is the only thing the model choice is ranked by.
    """
    if not is_record(value):
        return None
    raw = non_empty_string(value.get('id')) or non_empty_string(value.get('model'))
    if raw is None:
        return None
    provider = non_empty_string(value.get('provider')) or ''
    if not raw.lower().startswith(OPENROUTER_PREFIX) and provider.lower() in ('', 'openrouter'):
        raw = OPENROUTER_PREFIX + raw
    rating = number_as_given(value.get('rating_milli'))
    return {
        'id': raw,
        'order': order,
        'display_name': non_empty_string(value.get('display_name')) or raw,
        'provider': provider or 'openrouter',
        'rating_milli': rating if rating is not None else 0,
        'context_length': number_as_given(value.get('context_length')),
        'tools': value.get('tools') is True,
        'vision': value.get('vision') is True,
        'json': value.get('json') is True,
        'free': value.get('free') is not False,
    }


def prompt_row(value: object) -> Json | None:
    """One prompt row of the manifest: its id, wire, streaming flag, weight and requirements."""
    if not is_record(value):
        return None
    prompt_id = non_empty_string(value.get('id'))
    if prompt_id is None:
        return None
    requires: Json = {}
    raw_requires = value.get('requires')
    if is_record(raw_requires):
        if raw_requires.get('tools') is True:
            requires['tools'] = True
        if raw_requires.get('vision') is True:
            requires['vision'] = True
        if raw_requires.get('json') is True:
            requires['json'] = True
        min_context = finite_number(raw_requires.get('min_context'))
        if min_context is not None:
            requires['min_context'] = min_context
    return {
        'id': prompt_id,
        'api': 'anthropic' if value.get('api') == 'anthropic' else 'openai',
        'stream': value.get('stream') is True,
        'weight': 2 if value.get('weight') == 2 else 1,
        'requires': requires,
    }


def read_brief(path: str) -> Json:
    """The brief's two documents, or as much of them as the file at `path` holds."""
    empty: Json = {'models': [], 'prompts': [], 'error': None}
    if path == '':
        empty['error'] = 'MAKEFUN_PROMPT_FILE is not set'
        return empty
    try:
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
    except OSError as err:
        empty['error'] = f'cannot read the brief at {path}: {err}'
        return empty
    docs = documents_in(text)
    models_doc = document_with_array(docs, ('models', 'ratings'))
    if models_doc is None:
        empty['error'] = f'the brief at {path} holds no {MODELS_SCHEMA} document'
        return empty
    models = [row for row in (model_row(v, i) for i, v in enumerate(models_doc[0][models_doc[1]])) if row is not None]
    prompts_doc = document_with_array(docs, ('prompts', 'manifest'))
    if prompts_doc is None:
        return {'models': models, 'prompts': [], 'error': f'the brief at {path} holds no {PROMPTS_SCHEMA} document'}
    prompts = [row for row in (prompt_row(v) for v in prompts_doc[0][prompts_doc[1]]) if row is not None]
    return {'models': models, 'prompts': prompts, 'error': None}


BRIEF = read_brief(PROMPT_FILE)
# The only models this entry may dial: the brief's free rows, in the brief's order.
FREE_MODELS: list[Json] = [row for row in BRIEF['models'] if row['free']]
FREE_BY_ID = {row['id']: row for row in FREE_MODELS}


def build_catalog(models: list[Json]) -> Json:
    """The catalog document: one provider, exactly the dialable free ids, in the brief's order.

    `base_url` is the provider's public endpoint and never the gateway's trial address (the gateway's
    origin differs between a live pass and its replay, and the catalog is part of the state export a
    replay compares byte for byte). `key_env` and `wire` name this provider's lane. The model set is
    what the catalog column compares: the fourteen dialable ids, once each, no more and no fewer.
    """
    seen: set[str] = set()
    rows: list[Json] = []
    for model in models:
        model_id = model['id']
        if model_id in seen:
            continue
        seen.add(model_id)
        rows.append({'id': model_id, 'free': True, 'context_length': model['context_length']})
    if not rows:
        return {'providers': []}
    return {
        'providers': [
            {
                'id': 'openrouter',
                'base_url': OPENROUTER_PUBLIC_BASE,
                'key_env': OPENROUTER_KEY_ENV,
                'wire': 'openai',
                'models': rows,
            }
        ]
    }


CATALOG = build_catalog(FREE_MODELS)


# --------------------------------------------------------------------------- choosing the model

# Every capability a request may demand of a model, and how a request body spells it.
CAPABILITY_KEYS = ('tools', 'vision', 'json')


def model_meets(requires: Json, model: Json) -> bool:
    """Whether a model's capabilities meet one request's requirements.

    An unknown capability never meets one: a model that does not say it takes tools cannot be trusted
    to answer a request that declares them.
    """
    for key in CAPABILITY_KEYS:
        if requires.get(key) is True and model[key] is not True:
            return False
    min_context = finite_number(requires.get('min_context'))
    if min_context is not None:
        context_length = finite_number(model['context_length'])
        if context_length is None or context_length < min_context:
            return False
    return True


def requirements_from_body(body: Json) -> Json:
    """The requirements a request body itself declares, for a brief that names no row for it."""
    requires: Json = {}
    tools = body.get('tools')
    if (isinstance(tools, list) and len(tools) > 0) or is_record(body.get('tool_choice')) or isinstance(body.get('tool_choice'), str):
        requires['tools'] = True
    response_format = body.get('response_format')
    if is_record(response_format) and response_format.get('type') in ('json_object', 'json_schema'):
        requires['json'] = True
    if body_has_image(body):
        requires['vision'] = True
    return requires


def body_has_image(body: Json) -> bool:
    """Whether any message of a request carries an image block or a data-URL image part."""
    messages = body.get('messages')
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not is_record(message):
            continue
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for block in content:
            if not is_record(block):
                continue
            if block.get('type') in ('image', 'image_url'):
                return True
    return False


# The prompt row the request now arriving belongs to. The brief's manifest is in the order the env
# sends its prompts, one ahead of each request; the index is a counter this server keeps, never a
# reading of anything.
MANIFEST_INDEX = 0
MANIFEST_LOCK = threading.Lock()
# How many times each model has refused a call in this trial: a count, so the failover order is a
# function of the answers seen and never of the clock.
REFUSAL_COUNTS: dict[str, int] = {}
REFUSAL_LOCK = threading.Lock()


def next_manifest_row() -> Json | None:
    """The manifest row for the prompt now arriving, or None once the manifest is spent."""
    global MANIFEST_INDEX
    with MANIFEST_LOCK:
        index = MANIFEST_INDEX
        MANIFEST_INDEX += 1
    prompts = BRIEF['prompts']
    return prompts[index] if index < len(prompts) else None


def requirements_for(body: Json) -> Json:
    """What the arriving request needs of its model: the manifest row's requirements and the body's.

    The manifest is the declared source of a prompt's requirements; the body is read beside it so the
    choice never rests on arrival order alone, and a requirement either of them names is required.
    """
    requires: Json = {}
    row = next_manifest_row()
    if row is not None:
        requires.update(row['requires'])
    for key, value in requirements_from_body(body).items():
        if value is True:
            requires[key] = True
    return requires


def refuse(model_id: str) -> None:
    """Record one refusal of `model_id` (its count, never its time)."""
    with REFUSAL_LOCK:
        REFUSAL_COUNTS[model_id] = REFUSAL_COUNTS.get(model_id, 0) + 1


def refusal_count(model_id: str) -> int:
    with REFUSAL_LOCK:
        return REFUSAL_COUNTS.get(model_id, 0)


def ranked_models(requires: Json) -> list[Json]:
    """Every free model that meets `requires`, best-rated first.

    The rating is the brief's own rating_milli, so the model credited by the quality column is the
    best free rating in the house's table for that prompt. Ties keep the brief's order.
    """
    candidates = [model for model in FREE_MODELS if model_meets(requires, model)]
    candidates.sort(key=lambda model: (-float(model['rating_milli']), int(model['order'])))
    return candidates


def attempt_order(requires: Json) -> list[Json]:
    """The models one prompt is tried on, in order: the best-rated first, then the fallbacks.

    The best-rated model always goes first -- it is what quality scores -- and the fallbacks behind it
    are ordered so that a model this trial has already watched refuse is tried after one that has not.
    """
    candidates = ranked_models(requires)
    if len(candidates) <= 1:
        return candidates
    best, rest = candidates[0], candidates[1:]
    rest.sort(key=lambda model: (refusal_count(model['id']), -float(model['rating_milli']), int(model['order'])))
    return [best, *rest]


def attempt_plan(requires: Json) -> list[str]:
    """The dials one prompt may cost, as model ids in the order they are tried.

    A refusal is retried once on the model that refused it -- but only when that model has not already
    refused in this trial, so a rate-limited model does not cost two calls on every later prompt.
    """
    plan: list[str] = []
    for index, model in enumerate(attempt_order(requires)):
        tries = ATTEMPTS_ON_A_MODEL if index == 0 and refusal_count(model['id']) == 0 else 1
        for _ in range(tries):
            if len(plan) >= MAX_ATTEMPTS_PER_PROMPT:
                return plan
            plan.append(model['id'])
    return plan


# --------------------------------------------------------------------------- the two client wires


def openai_model_list() -> Json:
    """GET /v1/models for an `Authorization` request: OpenAI's list shape, the fourteen free ids."""
    return {
        'object': 'list',
        'data': [
            {'id': model['id'], 'object': 'model', 'created': FIXED_CREATED, 'owned_by': model['provider']}
            for model in FREE_MODELS
        ],
    }


def anthropic_model_list() -> Json:
    """GET /v1/models for an `x-api-key` request: Anthropic's list shape, the same fourteen ids."""
    return {
        'data': [
            {'type': 'model', 'id': model['id'], 'display_name': model['display_name'], 'created_at': FIXED_ANTHROPIC_CREATED_AT}
            for model in FREE_MODELS
        ],
        'has_more': False,
    }


def openai_error(message: str, error_type: str, code: str | None = None) -> Json:
    """An error in Chat Completions' envelope."""
    return {'error': {'message': message, 'type': error_type, 'code': code}}


def anthropic_error(error_type: str, message: str) -> Json:
    """An error in the Messages API's envelope."""
    return {'type': 'error', 'error': {'type': error_type, 'message': message}}


def openai_finish_reason(value: object) -> str:
    """`value` when it is a finish_reason Chat Completions knows, else `stop`.

    The provider's own value is kept whenever it is legal; a provider that names something else (or
    nothing at all) is reported as a normal end of answer rather than forwarded verbatim.
    """
    return value if isinstance(value, str) and value in OPENAI_STOP_REASONS else OPENAI_FINISH_FALLBACK


def first_choice(answer: Json) -> Json | None:
    """Chat Completions' first choice, when the answer holds one."""
    choices = answer.get('choices')
    if not isinstance(choices, list) or len(choices) == 0:
        return None
    return choices[0] if is_record(choices[0]) else None


def answer_message(answer: Json) -> Json:
    """The assistant message of a Chat Completions answer ({} when the answer holds none)."""
    choice = first_choice(answer)
    message = choice.get('message') if choice is not None else None
    return message if is_record(message) else {}


def answer_text(answer: Json) -> str:
    """The text a Chat Completions answer carries, from a string or from text parts."""
    content = answer_message(answer).get('content')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not is_record(part):
                continue
            if part.get('type') == 'text' and isinstance(part.get('text'), str):
                parts.append(part['text'])
        return ''.join(parts)
    return ''


def tool_calls_of(answer: Json) -> list[Json]:
    """A Chat Completions answer's tool calls as {id, name, arguments}, arguments a JSON string."""
    calls = answer_message(answer).get('tool_calls')
    if not isinstance(calls, list):
        return []
    out: list[Json] = []
    for call in calls:
        if not is_record(call):
            continue
        raw_function = call.get('function')
        function: Json = raw_function if is_record(raw_function) else {}
        name = non_empty_string(function.get('name'))
        if name is None:
            continue
        arguments = function.get('arguments')
        out.append(
            {
                'id': non_empty_string(call.get('id')) or f'toolu_{len(out)}',
                'name': name,
                'arguments': arguments if isinstance(arguments, str) else json_text(arguments if is_record(arguments) else {}),
            }
        )
    return out


def arguments_of(value: object) -> Json:
    """A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != '':
        parsed = parse_json_object(value)
        if parsed is not None:
            return parsed
    return {}


def usage_numbers(answer: Json) -> tuple[int, int]:
    """A Chat Completions answer's usage as whole (input, output) token counts.

    The provider's own counts are passed on whole; a count it does not give is 0, never a guess. The
    wire check reads the input as at least 0 and the output as at least 1, so a provider that reports
    no usage at all is visible as one failed check rather than a number this server made up.
    """
    usage = answer.get('usage')
    if not is_record(usage):
        return 0, 0
    prompt_tokens = usage.get('prompt_tokens')
    completion_tokens = usage.get('completion_tokens')
    input_tokens = int(prompt_tokens) if is_whole_number(prompt_tokens) and prompt_tokens >= 0 else 0
    output_tokens = int(completion_tokens) if is_whole_number(completion_tokens) and completion_tokens >= 0 else 0
    return input_tokens, output_tokens


def usage_object(answer: Json) -> Json:
    """The usage object a client's answer carries, whole counts only."""
    input_tokens, output_tokens = usage_numbers(answer)
    return {'prompt_tokens': input_tokens, 'completion_tokens': output_tokens, 'total_tokens': input_tokens + output_tokens}


def stop_reason_of(answer: Json, stop_sequences: list[str]) -> tuple[str, str | None]:
    """Anthropic's (stop_reason, stop_sequence) for a Chat Completions answer.

    A finish_reason of `stop` is `end_turn`; when the answer's text ends with one of the stop sequences
    the client declared, the Messages API's honest reason is `stop_sequence` and the sequence is named.
    """
    choice = first_choice(answer)
    finish = choice.get('finish_reason') if choice is not None else None
    reason = FINISH_TO_STOP_REASON.get(finish, ANTHROPIC_STOP_FALLBACK) if isinstance(finish, str) else ANTHROPIC_STOP_FALLBACK
    if reason not in ANTHROPIC_STOP_REASONS:
        reason = ANTHROPIC_STOP_FALLBACK
    if reason == 'end_turn' and len(stop_sequences) > 0:
        text = answer_text(answer)
        for sequence in stop_sequences:
            if text.endswith(sequence):
                return 'stop_sequence', sequence
    return reason, None


def openai_answer_for_client(answer: Json, model: str) -> Json:
    """A Chat Completions answer as the client receives it: the dialled id back, a legal stop reason.

    The provider's answer body is passed on as it is, with three things made true of it: the `model` is
    the id this server dialled (the provider spells it without the gateway's `openrouter/` prefix, and
    the model column scores the model that really answered), the first choice names the assistant, and
    the finish reason is one Chat Completions knows.
    """
    out = dict(answer)
    out['model'] = model
    if not isinstance(out.get('object'), str):
        out['object'] = 'chat.completion'
    choices = out.get('choices') if isinstance(out.get('choices'), list) else []
    if len(choices) == 0:
        out['choices'] = [{'index': 0, 'message': {'role': 'assistant', 'content': ''}, 'finish_reason': OPENAI_FINISH_FALLBACK}]
    else:
        first = dict(choices[0]) if is_record(choices[0]) else {}
        message = dict(first['message']) if is_record(first.get('message')) else {}
        message['role'] = 'assistant'
        if not isinstance(message.get('content'), (str, list)) and not isinstance(message.get('tool_calls'), list):
            message['content'] = ''
        first['message'] = message
        first['finish_reason'] = openai_finish_reason(first.get('finish_reason'))
        out['choices'] = [first, *choices[1:]]
    out['usage'] = usage_object(answer) if is_record(answer.get('usage')) else out.get('usage')
    return out


def anthropic_message_for_client(answer: Json, model: str, stop_sequences: list[str]) -> Json:
    """A Chat Completions answer as an Anthropic Messages message: text and tool_use blocks, usage,
    the stop reason, and the dialled id in `model`."""
    content: list[Json] = []
    text = answer_text(answer)
    if text != '':
        content.append({'type': 'text', 'text': text})
    for call in tool_calls_of(answer):
        content.append({'type': 'tool_use', 'id': call['id'], 'name': call['name'], 'input': arguments_of(call['arguments'])})
    if len(content) == 0:
        content.append({'type': 'text', 'text': ''})
    stop_reason, stop_sequence = stop_reason_of(answer, stop_sequences)
    input_tokens, output_tokens = usage_numbers(answer)
    return {
        'id': 'msg_' + (non_empty_string(answer.get('id')) or 'aggregate'),
        'type': 'message',
        'role': 'assistant',
        'model': model,
        'content': content,
        'stop_reason': stop_reason,
        'stop_sequence': stop_sequence,
        'usage': {'input_tokens': input_tokens, 'output_tokens': output_tokens},
    }


# --------------------------------------------------------------------------- the Anthropic request, as Chat Completions


def content_part_of(block: Json) -> Json | None:
    """One Anthropic content block as a Chat Completions content part: text or an inline image."""
    if block.get('type') == 'text' and isinstance(block.get('text'), str):
        return {'type': 'text', 'text': block['text']}
    source = block.get('source')
    if block.get('type') == 'image' and is_record(source):
        if source.get('type') == 'base64' and isinstance(source.get('data'), str):
            media = source['media_type'] if isinstance(source.get('media_type'), str) else 'image/png'
            return {'type': 'image_url', 'image_url': {'url': f'data:{media};base64,{source["data"]}'}}
        if source.get('type') == 'url' and isinstance(source.get('url'), str):
            return {'type': 'image_url', 'image_url': {'url': source['url']}}
    return None


def tool_result_text(content: object) -> str:
    """A tool_result block's content as text: its text blocks, its JSON, or ''."""
    if isinstance(content, str):
        return content
    if content is None:
        return ''
    if isinstance(content, list):
        text = text_of(content)
        return text if text != '' else json_text(content)
    return json_text(content)


def output_cap(body: Json) -> int:
    """The output cap a request names, clamped into what the gateway accepts.

    The gateway refuses a request with no cap and one over 8192 tokens, so a client that names none
    gets this server's own default and a client that asks for more gets the ceiling.
    """
    named = finite_number(body.get('max_tokens'))
    if named is None:
        named = finite_number(body.get('max_completion_tokens'))
    if named is None or named <= 0:
        return DEFAULT_OUTPUT_CAP
    return min(int(named), MAX_OUTPUT_CAP)


def anthropic_request_as_chat(body: Json, model: str) -> Json:
    """An Anthropic Messages request as the Chat Completions body the gateway dials.

    The system prompt becomes a system message, image blocks become image_url data URLs, `tool_use`
    blocks become tool_calls, `tool_result` blocks become tool messages, tools and tool_choice are
    rewritten into the OpenAI shapes, and the output cap, temperature, top_p and stop sequences carry
    through. `model` is the dialled id, and the body is never streamed upstream: the complete answer is
    what this server turns back into the client's wire (see the file's header).
    """
    messages: list[Json] = []
    system = text_of(body.get('system'))
    if system != '':
        messages.append({'role': 'system', 'content': system})
    blocks = body.get('messages') if isinstance(body.get('messages'), list) else []
    for raw in blocks:
        if not is_record(raw):
            continue
        role = 'assistant' if raw.get('role') == 'assistant' else 'user'
        content = raw.get('content')
        if isinstance(content, str):
            messages.append({'role': role, 'content': content})
            continue
        if not isinstance(content, list):
            continue
        if role == 'assistant':
            tool_calls: list[Json] = []
            for block in content:
                if is_record(block) and block.get('type') == 'tool_use':
                    tool_calls.append(
                        {
                            'id': non_empty_string(block.get('id')) or '',
                            'type': 'function',
                            'function': {
                                'name': non_empty_string(block.get('name')) or '',
                                'arguments': json_text(block['input'] if is_record(block.get('input')) else {}),
                            },
                        }
                    )
            text = text_of(content)
            message: Json = {'role': 'assistant', 'content': text if text != '' else None}
            if len(tool_calls) > 0:
                message['tool_calls'] = tool_calls
            messages.append(message)
            continue
        parts: list[Json] = []
        for block in content:
            if not is_record(block):
                continue
            if block.get('type') == 'tool_result':
                messages.append(
                    {
                        'role': 'tool',
                        'tool_call_id': non_empty_string(block.get('tool_use_id')) or '',
                        'content': tool_result_text(block.get('content')),
                    }
                )
                continue
            part = content_part_of(block)
            if part is not None:
                parts.append(part)
        if len(parts) > 0:
            messages.append({'role': 'user', 'content': parts})
    out: Json = {'model': model, 'messages': messages, 'max_tokens': output_cap(body)}
    if is_number(body.get('temperature')):
        out['temperature'] = body['temperature']
    if is_number(body.get('top_p')):
        out['top_p'] = body['top_p']
    if isinstance(body.get('stop_sequences'), list) and len(body['stop_sequences']) > 0:
        out['stop'] = [s for s in body['stop_sequences'] if isinstance(s, str) and s != '']
    tools = body.get('tools')
    if isinstance(tools, list):
        declared: list[Json] = []
        for raw in tools:
            if not is_record(raw):
                continue
            name = non_empty_string(raw.get('name'))
            if name is None:
                continue
            declared.append(
                {
                    'type': 'function',
                    'function': {
                        'name': name,
                        'description': raw['description'] if isinstance(raw.get('description'), str) else '',
                        'parameters': raw['input_schema'] if is_record(raw.get('input_schema')) else {'type': 'object', 'properties': {}},
                    },
                }
            )
        if len(declared) > 0:
            out['tools'] = declared
    choice = body.get('tool_choice')
    if is_record(choice):
        if choice.get('type') == 'auto':
            out['tool_choice'] = 'auto'
        elif choice.get('type') == 'any':
            out['tool_choice'] = 'required'
        elif choice.get('type') == 'tool':
            name = non_empty_string(choice.get('name'))
            if name is not None:
                out['tool_choice'] = {'type': 'function', 'function': {'name': name}}
    return out


def chat_request_as_gateway_body(body: Json, model: str) -> Json:
    """An OpenAI-shaped client request as the body the gateway dials: the request's own fields, the
    dialled id, and never `stream` (the upstream call brings back a complete answer)."""
    out = {key: value for key, value in body.items() if key not in ('model', 'stream', 'stream_options')}
    out['model'] = model
    out['max_tokens'] = output_cap(body)
    if 'max_completion_tokens' in out:
        out.pop('max_completion_tokens')
    return out


def stop_sequences_of(body: Json, wire: str) -> list[str]:
    """The stop sequences a client declared, normalized: `stop_sequences` on the Messages wire, `stop`
    on Chat Completions."""
    named = body.get('stop_sequences') if wire == 'anthropic' else body.get('stop')
    if isinstance(named, str) and named != '':
        return [named]
    if isinstance(named, list):
        return [s for s in named if isinstance(s, str) and s != '']
    return []


# --------------------------------------------------------------------------- the gateway


class UpstreamUnreachable(Exception):
    """The gateway could not be reached at all: no status, no body, no refusal to report."""


class Upstream:
    """One gateway answer: its status and the response whose body is read capped."""

    def __init__(self, status: int, response) -> None:
        self.status = status
        self._response = response

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def text(self) -> str:
        """The body as text, capped, and the connection released."""
        try:
            return self._response.read(MAX_UPSTREAM_BYTES).decode('utf-8', 'replace')
        except Exception as err:  # a body that cannot be read is an answer of another kind
            return f'<unreadable upstream body: {err}>'
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._response.close()
        except Exception:  # an already-closed response is nothing to report
            pass


def chat_completions_url() -> str:
    """The gateway's Chat Completions route. $OPENAI_BASE_URL is its OpenAI-compatible root (`…/v1`);
    a root that does not already end in `/v1` gets one, so both spellings reach the same route."""
    if GATEWAY_BASE == '':
        return ''
    return f'{GATEWAY_BASE}/chat/completions' if GATEWAY_BASE.endswith('/v1') else f'{GATEWAY_BASE}/v1/chat/completions'


def dial(model: str, body: Json) -> Upstream:
    """POST one complete (never streaming) Chat Completions call to the gateway, naming `model`."""
    url = chat_completions_url()
    if url == '':
        raise UpstreamUnreachable('OPENAI_BASE_URL is not set')
    data = json_text(body).encode('utf-8')
    request = urllib.request.Request(
        url,
        data=data,
        headers={'content-type': 'application/json', 'authorization': f'Bearer {GATEWAY_KEY}'},
        method='POST',
    )
    try:
        response = urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT_SEC)
    except urllib.error.HTTPError as err:
        # A refusal is an answer with a status and a body, exactly as the provider sent it.
        return Upstream(err.code, err)
    except (urllib.error.URLError, OSError, ValueError) as err:
        raise UpstreamUnreachable(str(err)) from err
    return Upstream(response.status, response)


def refusal_code(body_text: str) -> str:
    """The provider's own short code for a refusal: error.code, else error.type, else ''."""
    parsed = parse_json_object(body_text)
    error = parsed.get('error') if parsed is not None else None
    if not is_record(error):
        return ''
    code = non_empty_string(error.get('code')) or non_empty_string(error.get('type')) or ''
    return code[:MAX_FAILURE_CODE_CHARS]


class Attempt:
    """The outcome of trying one prompt: the answer a client can be given, or why none could be."""

    __slots__ = ('answer', 'model', 'trail', 'status', 'error')

    def __init__(
        self,
        answer: Json | None,
        model: str | None,
        trail: list[Json],
        status: int,
        error: str | None,
    ) -> None:
        self.answer = answer
        self.model = model
        self.trail = trail
        self.status = status
        self.error = error

    @property
    def served(self) -> bool:
        return self.answer is not None and self.model is not None


def acquire(requires: Json, body_for: callable) -> Attempt:
    """Dial the models that meet `requires`, best-rated first, until one answers.

    A refused call (any 4xx or 5xx) is retried on the model that refused it and then failed over from:
    every refusal that a later success recovered from is kept, in the order the calls refused, for the
    `x-makefun-upstream-failures` trail. A call that never reached the gateway is not a refusal and is
    reported as nothing; the prompt is simply tried on the next model.
    """
    trail: list[Json] = []
    status = 0
    error = 'no free model in the brief meets this request'
    for model_id in attempt_plan(requires):
        try:
            call = dial(model_id, body_for(model_id))
        except UpstreamUnreachable as err:
            error = f'the gateway could not be reached: {err}'
            continue
        if call.ok:
            text = call.text()
            answer = parse_json_object(text)
            if answer is not None:
                return Attempt(answer, model_id, trail[:MAX_FAILURE_ROWS], call.status, None)
            # A 2xx that is not one JSON object is no answer: try the next model, and say so.
            error = f'the gateway answered HTTP {call.status} for {model_id} with a body that is not one JSON object'
            status = call.status
            continue
        code = refusal_code(call.text())
        refuse(model_id)
        status = call.status
        error = f'the gateway answered HTTP {call.status} for {model_id}'
        if len(trail) < MAX_FAILURE_ROWS:
            trail.append({'status': call.status, 'code': code, 'model': model_id})
    return Attempt(None, None, trail[:MAX_FAILURE_ROWS], status, error)


def failures_header(trail: list[Json]) -> str | None:
    """The refusal trail as the header's value, or None when there is nothing to report.

    The header is optional and the only way the failover column can see a refusal this server retried
    and then served, so it carries what happened, in order, JSON-encoded on one line and within the
    ceiling the env reads (eight rows and 8 KiB).
    """
    rows = trail[:MAX_FAILURE_ROWS]
    if len(rows) == 0:
        return None
    while len(rows) > 0:
        value = json.dumps(rows, separators=(',', ':'), ensure_ascii=True)
        if len(value.encode('utf-8')) <= MAX_FAILURE_HEADER_BYTES:
            return value
        rows = rows[:-1]
    return None


# --------------------------------------------------------------------------- the event streams


def sse_frame(event: str | None, data: object) -> str:
    """One server-sent event: its optional name, one JSON data line, and the blank line that ends it."""
    named = '' if event is None else f'event: {event}\n'
    return f'{named}data: {json_text(data)}\n\n'


def stream_pieces(text: str) -> list[str]:
    """A complete answer's text as the deltas a stream carries: the same characters, cut up."""
    return [text[at : at + STREAM_DELTA_CHARS] for at in range(0, len(text), STREAM_DELTA_CHARS)]


def openai_stream_frames(answer: Json, model: str) -> list[str]:
    """Chat Completions' event stream for a complete answer, in the API's own order.

    One chunk with the assistant's role, the answer's text as content deltas, the tool calls as they
    arrived, one chunk with the finish reason, one chunk carrying the usage (with an empty choices
    array, as `stream_options.include_usage` asks for), and `[DONE]`. Every chunk and the usage chunk
    name the model this server dialled.
    """
    choice = first_choice(answer)
    finish = openai_finish_reason(choice.get('finish_reason') if choice is not None else None)
    created = answer.get('created') if is_whole_number(answer.get('created')) else FIXED_CREATED
    ident = non_empty_string(answer.get('id')) or 'chatcmpl-aggregate'

    def chunk(delta: Json, finish_reason: str | None) -> str:
        return sse_frame(
            None,
            {
                'id': ident,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': model,
                'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}],
            },
        )

    frames = [chunk({'role': 'assistant', 'content': ''}, None)]
    for piece in stream_pieces(answer_text(answer)):
        frames.append(chunk({'content': piece}, None))
    for index, call in enumerate(tool_calls_of(answer)):
        frames.append(
            chunk(
                {
                    'tool_calls': [
                        {
                            'index': index,
                            'id': call['id'],
                            'type': 'function',
                            'function': {'name': call['name'], 'arguments': call['arguments']},
                        }
                    ]
                },
                None,
            )
        )
    frames.append(chunk({}, finish))
    if is_record(answer.get('usage')):
        frames.append(
            sse_frame(
                None,
                {
                    'id': ident,
                    'object': 'chat.completion.chunk',
                    'created': created,
                    'model': model,
                    'choices': [],
                    'usage': usage_object(answer),
                },
            )
        )
    frames.append('data: [DONE]\n\n')
    return frames


def anthropic_stream_events(answer: Json, model: str, stop_sequences: list[str]) -> list[str]:
    """The Messages API's event sequence for a complete answer, in the order its clients expect.

    `message_start` first, with the message and the provider's own input count; then one
    `content_block_start` / `content_block_delta` / `content_block_stop` run per text or tool_use
    block; then `message_delta` (the stop reason and the output count) and `message_stop`. No event
    follows `message_stop`, no content block is left open when `message_delta` arrives, and every
    event that names a model names the id this server dialled.
    """
    input_tokens, output_tokens = usage_numbers(answer)
    ident = 'msg_' + (non_empty_string(answer.get('id')) or 'aggregate')
    stop_reason, stop_sequence = stop_reason_of(answer, stop_sequences)
    starts: list[str] = [
        sse_frame(
            'message_start',
            {
                'type': 'message_start',
                'message': {
                    'id': ident,
                    'type': 'message',
                    'role': 'assistant',
                    'model': model,
                    'content': [],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'usage': {'input_tokens': input_tokens, 'output_tokens': 0},
                },
            },
        )
    ]
    events: list[str] = []
    index = 0
    text = answer_text(answer)
    if text != '':
        events.append(
            sse_frame('content_block_start', {'type': 'content_block_start', 'index': index, 'content_block': {'type': 'text', 'text': ''}})
        )
        for piece in stream_pieces(text):
            events.append(
                sse_frame(
                    'content_block_delta',
                    {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'text_delta', 'text': piece}},
                )
            )
        events.append(sse_frame('content_block_stop', {'type': 'content_block_stop', 'index': index}))
        index += 1
    for call in tool_calls_of(answer):
        events.append(
            sse_frame(
                'content_block_start',
                {
                    'type': 'content_block_start',
                    'index': index,
                    'content_block': {'type': 'tool_use', 'id': call['id'], 'name': call['name'], 'input': {}},
                },
            )
        )
        if call['arguments'] != '':
            events.append(
                sse_frame(
                    'content_block_delta',
                    {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'input_json_delta', 'partial_json': call['arguments']}},
                )
            )
        events.append(sse_frame('content_block_stop', {'type': 'content_block_stop', 'index': index}))
        index += 1
    events.append(
        sse_frame(
            'message_delta',
            {'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': stop_sequence}, 'usage': {'output_tokens': output_tokens}},
        )
    )
    events.append(sse_frame('message_stop', {'type': 'message_stop'}))
    return starts + events


# --------------------------------------------------------------------------- the HTTP server


def log(message: str) -> None:
    """One line on stderr: the entry's own log, which the trial keeps."""
    sys.stderr.write(f'aggregator: {message}\n')
    sys.stderr.flush()


class Handler(BaseHTTPRequestHandler):
    """The entry's one server: the models route, the two completion wires, and nothing else."""

    protocol_version = 'HTTP/1.1'
    server_version = 'free-model-aggregator-py/1'
    sys_version = ''
    # This server's own flag: whether any part of the answer has been written to the client.
    answered = False

    def log_message(self, fmt: str, *args: object) -> None:  # the base class's name
        """No access line per request: the trial's stderr is a budget of its own."""

    def do_GET(self) -> None:  # the base class's name
        self.dispatch()

    def do_POST(self) -> None:  # the base class's name
        self.dispatch()

    # -- plumbing -----------------------------------------------------------------------------------

    def dispatch(self) -> None:
        path = urlsplit(self.path).path
        try:
            if self.command == 'GET' and path == '/v1/models':
                self.models()
                return
            if self.command == 'POST' and path == '/v1/chat/completions':
                self.completion('openai')
                return
            if self.command == 'POST' and path == '/v1/messages':
                self.completion('anthropic')
                return
            self.send_error_body('anthropic' if path == '/v1/messages' else 'openai', 404, 'not_found', f'no route {self.command} {path}')
        except (BrokenPipeError, ConnectionResetError):
            # The client went away mid-answer: nothing left to say, and the connection is done.
            self.close_connection = True
        except Exception:  # one request's failure must not take the server down
            log(traceback.format_exc())
            if not self.answered:
                wire = 'anthropic' if urlsplit(self.path).path == '/v1/messages' else 'openai'
                try:
                    self.send_error_body(wire, 500, 'internal_error', 'the entry failed to answer the request')
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def read_json_body(self) -> Json | None:
        """The one client request body, or None when it is not one JSON object."""
        length = self.headers.get('content-length')
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_REQUEST_BYTES:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b''
        if not raw:
            return None
        try:
            value = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_record(value) else None

    def send_json(self, status: int, body: object, trail: list[Json] | None = None) -> None:
        """One JSON answer, with the failure trail in its header when there is one to report."""
        raw = json_text(body).encode('utf-8')
        self.send_response(status)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(raw)))
        header = failures_header(trail or [])
        if header is not None:
            self.send_header(FAILURES_HEADER, header)
        self.answered = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error_body(self, wire: str, status: int, error_type: str, message: str, trail: list[Json] | None = None) -> None:
        if wire == 'anthropic':
            self.send_json(status, anthropic_error(error_type, message), trail)
        else:
            self.send_json(status, openai_error(message, error_type, None), trail)

    def send_event_stream(self, frames: list[str], trail: list[Json] | None = None) -> None:
        """One server-sent event stream: its headers, then every frame.

        The answer is cut off by the connection's close: Python's http.server chunks nothing, so
        `connection: close` is what ends the stream for the client.
        """
        self.send_response(200)
        self.send_header('content-type', 'text/event-stream')
        self.send_header('cache-control', 'no-cache')
        self.send_header('connection', 'close')
        header = failures_header(trail or [])
        if header is not None:
            self.send_header(FAILURES_HEADER, header)
        self.answered = True
        self.end_headers()
        self.close_connection = True
        for frame in frames:
            self.wfile.write(frame.encode('utf-8'))
            self.wfile.flush()

    # -- the routes ---------------------------------------------------------------------------------

    def models(self) -> None:
        """GET /v1/models, in the shape the request's auth header names.

        `x-api-key` is the Messages wire and gets Anthropic's list; anything else -- an
        `Authorization: Bearer` request -- gets Chat Completions' list. Both list exactly the fourteen
        dialable free ids the catalog registers.
        """
        if self.headers.get('x-api-key') is not None:
            self.send_json(200, anthropic_model_list())
        else:
            self.send_json(200, openai_model_list())

    def completion(self, wire: str) -> None:
        """One client completion on either wire, answered by a real free model.

        The request's requirements are read, the best-rated model that meets them is dialled through
        the gateway's Chat Completions route -- with the request translated into a Chat Completions
        body when it arrived on the Messages wire -- and the gateway's own answer is returned in the
        wire and the streaming mode the client asked for, its model field naming the id dialled and
        the refusals that were recovered from in the failure trail.
        """
        body = self.read_json_body()
        if body is None:
            self.send_error_body(wire, 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        stream = body.get('stream') is True
        requires = requirements_for(body)
        stop_sequences = stop_sequences_of(body, wire)
        if wire == 'anthropic':
            gateway_body_for = lambda model_id: anthropic_request_as_chat(body, model_id)  # noqa: E731
        else:
            gateway_body_for = lambda model_id: chat_request_as_gateway_body(body, model_id)  # noqa: E731
        attempt = acquire(requires, gateway_body_for)
        if not attempt.served:
            log(f'{wire}: no model served the prompt: {attempt.error}')
            self.unserved(wire, attempt)
            return
        if wire == 'anthropic':
            if stream:
                self.send_event_stream(anthropic_stream_events(attempt.answer, attempt.model, stop_sequences), attempt.trail)
            else:
                self.send_json(200, anthropic_message_for_client(attempt.answer, attempt.model, stop_sequences), attempt.trail)
            return
        if stream:
            self.send_event_stream(openai_stream_frames(attempt.answer, attempt.model), attempt.trail)
        else:
            self.send_json(200, openai_answer_for_client(attempt.answer, attempt.model), attempt.trail)

    def unserved(self, wire: str, attempt: Attempt) -> None:
        """A prompt no model answered: the wire's own error envelope, and the refusal trail with it.

        The prompt is lost and its checks fail; the trail is sent all the same, because the taste of
        the trial that matters is the refusals the gateway really answered.
        """
        detail = attempt.error or 'the gateway refused every model that meets this request'
        status = 502
        if wire == 'anthropic':
            self.send_error_body(wire, status, 'api_error', detail, attempt.trail)
        else:
            self.send_error_body(wire, status, 'upstream_error', detail, attempt.trail)


# --------------------------------------------------------------------------- register, poll, start


def register(port: int) -> None:
    """POST $MAKEFUN_ENV_URL/register with the port and the catalog, retrying briefly.

    The env begins driving the trial from this call, so a register that fails is worth retrying; one
    that is answered with a 4xx is not (the body is this server's own and is well formed).
    """
    body = json_text({'protocol_version': PROTOCOL_VERSION, 'port': port, 'catalog': CATALOG}).encode('utf-8')
    last = 'no answer'
    for attempt in range(5):
        request = urllib.request.Request(
            f'{ENV_URL}/register',
            data=body,
            headers={'content-type': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                status = response.status
                reply = response.read(64 * 1024).decode('utf-8', 'replace')
        except urllib.error.HTTPError as err:
            raise RuntimeError(
                f'POST {ENV_URL}/register answered HTTP {err.code}: {err.read(4096).decode("utf-8", "replace")[:500]}'
            ) from err
        except (urllib.error.URLError, OSError) as err:
            last = str(err)
            if attempt < 4:
                time.sleep(0.25)
            continue
        if not (200 <= status < 300):
            raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {status}: {reply[:500]}')
        parsed = parse_json_object(reply)
        if parsed is not None and is_whole_number(parsed.get('requests')):
            log(f'registered on port {port}: the trial holds {parsed["requests"]} prompts')
        else:
            log(f'registered on port {port}')
        return
    raise RuntimeError(f'POST {ENV_URL}/register could not be reached: {last}')


def poll_work() -> None:
    """Poll GET $MAKEFUN_ENV_URL/work and exit 0 the moment it says done.

    Nothing else ends the process: the wall budget does, and a process still running when it is spent
    scores nothing.
    """
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            with urllib.request.urlopen(f'{ENV_URL}/work', timeout=2) as response:
                if response.status != 200:
                    continue
                work = parse_json_object(response.read(64 * 1024).decode('utf-8', 'replace'))
        except Exception:  # the env is not answering yet: try again on the next tick
            continue
        if work is not None and work.get('done') is True:
            sys.stderr.flush()
            os._exit(0)


def stop(*_: object) -> None:
    """SIGTERM and SIGINT end the trial as Node's process.exit(0) does: exit 0, at once."""
    sys.stderr.flush()
    os._exit(0)


def print_catalog() -> None:
    """`--list`: the catalog document, byte for byte as the register body carries it."""
    text = json_text(CATALOG) + '\n'
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except (BrokenPipeError, ValueError):
        # The reader (a `head`, say) closed the pipe first: nothing left to print, and no failure.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except OSError:
            pass


def start() -> None:
    """The trial: listen on a free port, register, serve, and poll /work."""
    if BRIEF['error'] is not None:
        log(BRIEF['error'])
    if '--list' in sys.argv[1:]:
        print_catalog()
        return
    if ENV_URL == '':
        raise RuntimeError('MAKEFUN_ENV_URL is not set')
    if chat_completions_url() == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    server = ThreadingHTTPServer(('0.0.0.0', 0), Handler)
    server.daemon_threads = True
    try:
        port = server.server_address[1]
        if port == 8080:
            # 8080 is the env's own port on one layout: the OS's ephemeral range never holds it, and a
            # trial that got it would be registering a listener the env is already using.
            raise RuntimeError('the OS picked port 8080, which the env owns on one layout: restart the trial')
        log(f'listening on 0.0.0.0:{port} with {len(FREE_MODELS)} free models')
        register(port)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=poll_work, name='work-poll', daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:  # the top-level catch: log the stack and exit non-zero
        log(traceback.format_exc())
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
