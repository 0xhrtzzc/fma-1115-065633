# fma-1115-065633

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/0xhrtzzc/fma-1115-065633/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/0xhrtzzc/fma-1115-065633/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server that answers both completion wires the grader dials — OpenAI's Chat Completions and Anthropic's Messages — in JSON and in server-sent events, choosing for every prompt a free model the brief allows: by the ratings table, by what the prompt needs (strict JSON, tool calling, long context), and by failing over to the next-best rated model when one refuses, with every refusal reported in the answer's own failure header.

## Run it

```sh
python3 main.py   # serves OpenAI and Anthropic wires on one port, models dialled through the gateway in $GATEWAY_URL (or OPENAI_BASE_URL) with the key in $GATEWAY_KEY (or OPENAI_API_KEY)
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=/api/competitions/free-model-aggregator-02/submissions&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
