# What I verified about this server

I ran the entry locally against a stub env and a stub gateway (nothing here is the grader), so this note
says exactly which claims I watched hold and which I did not.

## Verified by running it

- `python3 -c "import ast;ast.parse(open('main.py').read())"` prints `parses ok`; the file is 1525 lines
  before the closing comment and uses only the standard library.
- `python3 -P -s main.py --list` exits 0 and prints byte for byte the same catalog document the register
  body carried (`json.dumps` with `separators=(',', ':')` plus a newline).
- After `POST /register`, the body carried `protocol_version: 1`, an integer port, and one provider whose
  model list was exactly the 14 free rows of the brief, once each — and whose `base_url` was the
  provider's public endpoint, with the trial-local gateway origin nowhere in the catalog.
- Both `GET /v1/models` shapes answered: `Authorization` gave OpenAI's list, `x-api-key` gave Anthropic's
  list, each with the same 14 ids.
- Failing over worked on a live refusal: with the top-rated free model answering HTTP 429 with
  `code: rate_limited`, the entry retried it once, then served the prompt from the next-best free model,
  and the `x-makefun-upstream-failures` header carried the two refused rows in order, JSON on one line.
- The model choice followed the brief, not a hardcoded id: a prompt needing JSON mode and 30k context
  went to a model that has both, and a prompt needing tools went to a tools-capable one.
- Both wires answered from the gateway's own completion: JSON and `text/event-stream` on
  `/v1/chat/completions` (role chunk, content deltas whose characters rejoin to the whole text, the usage
  chunk with an empty `choices`, then `[DONE]`), and JSON plus the `message_start / content_block_start /
  content_block_delta / content_block_stop / message_delta / message_stop` order on `/v1/messages`.
- Usage was passed on whole: `prompt_tokens 42` and `completion_tokens 7` came back as
  `total_tokens 49` on the OpenAI wire and as `input_tokens 42 / output_tokens 7` on the Messages wire,
  with no invented number.
- Stop reasons were legal on both wires (`stop`, and `tool_use` when the answer carried a tool call);
  tool calls survived the translation as `tool_use` blocks with their arguments parsed into an object.
- Every upstream call was non-streaming (no `stream` key in any dialled body), every dialled id was one
  of the brief's free ids, and the gateway was dialled with the trial's own key.
- `GET $MAKEFUN_ENV_URL/work` returning `done: true` made the process exit 0, and it logged its listen
  line to stderr on the way.

## Not verified here

- A real trial: the grader's env, its six prompts, its replay, and the real OpenRouter gateway were not
  available to me. My stub refused one model; the real upstream's refusals, latency and bodies were never
  seen.
- The replay comparison itself: I read the file for clock, randomness and UUID reads and found none, and
  the catalog is built from the brief alone, but I did not diff a live export against a replayed one.
- The eight-row / 8 KiB ceiling on the failure trail, and the prompt-is-spent case where the manifest runs
  out, were reasoned about from the code rather than exercised.

<!-- End of note: this file records what was checked by running the server, and what was not. -->
