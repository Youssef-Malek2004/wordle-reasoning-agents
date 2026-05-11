# Wordle Agents: Comparing Reasoning Strategies on a Constraint-Satisfaction Game

A small experiment in agent design: build four LLM-powered Wordle players using different reasoning strategies, run them on the same set of words, and see which approach actually helps. The interesting finding turned out not to be "which strategy wins" but **how poorly more reasoning helps when the underlying problem is mostly mechanical bookkeeping**.

Built with **LangGraph** for the game loop, **LangChain** for prompt orchestration, and **Google Gemma 4 26B-A4B-IT** via **OpenRouter** as the underlying LLM.

---

## The Problem

Wordle is a 5-letter-word guessing game with deterministic feedback per guess: each letter is **GREEN** (right letter, right place), **YELLOW** (right letter, wrong place), or **GREY** (not in the answer). The player has 6 guesses. The interesting question for an LLM agent isn't *can it guess words* — it's *can it correctly track and apply constraints across multiple turns*.

That single-line description hides the real challenge: by guess five, the model is juggling 25 colored cells of feedback and needs to propose a word consistent with every one of them. This is closer to a constraint-satisfaction problem than a language problem.

## The Four Agents

All four agents share the same LangGraph state machine — a single `guess` node that produces a candidate word, followed by a `feedback` node that scores it against the answer and either ends the game or loops back. What changes between agents is **how the `guess` node decides what to play**.

| Agent | Strategy | Calls per turn |
|---|---|---|
| **Baseline** | Single LLM call, JSON-structured output, no reasoning | 1 |
| **Chain-of-Thought** | LLM reasons out loud about confirmed/misplaced/eliminated letters, then commits to a guess on a final `GUESS:` line | 1 |
| **Self-Reflection** | Propose → critique → re-propose loop. A second LLM call critiques the candidate against the constraints; if INVALID, the proposer tries again with the critique fed back in (up to 3 attempts) | 2-6 |
| **Self-Consistency** | Sample 3 candidates at high temperature, take the majority vote, fall back to a heuristic constraint-satisfaction score on ties | 3 |

The architecture lives in factory functions (`make_baseline_node(llm)`, `make_cot_node(llm)`, etc.) so the same code runs against different model configurations by just swapping the LLM.

## Architecture

```
START -> guess -> feedback -> [won?  -> END]
                          \-> [lost? -> END]
                          \-> [continue -> guess]
```

State carried between nodes:

```python
class GameState(TypedDict):
    word: str                    # the answer (hidden from the LLM)
    guesses: list[Guess]         # full history with per-letter colors
    remaining_guesses: int
```

The `feedback_node` is a pure function — it scores the most recent guess against the answer using `score_guess`, which correctly handles Wordle's duplicate-letter rules (a letter can be GREY in one position while GREEN/YELLOW in another when the guess has more of that letter than the answer). This same scorer could in principle replace the LLM critic in the Self-Reflection agent — see the reflection in the notebook for why I didn't.

## Running It

You'll need an OpenRouter API key (the notebook also has commented-out blocks for direct Gemini and a local mlx server, in case you want to swap providers).

```bash
# 1. Clone and enter
git clone https://github.com/Youssef-Malek2004/genai-assignment-2.git
cd genai-assignment-2

# 2. Set up the API key
cp .env.example .env
# Edit .env and put your OpenRouter key in OPENROUTER_API_KEY

# 3. Install deps (a conda env or venv is fine)
pip install langgraph langchain-core langchain-openai langchain-google-genai \
            langchain-openrouter pydantic python-dotenv typing_extensions

# 4. Open the notebook
jupyter notebook genai-2-assignment.ipynb
```

Run the cells top to bottom. Each agent is in its own experiment cell so you can run them independently.

## Findings

Real results vary run-to-run because the model is non-deterministic, but the patterns held across multiple runs. The honest summary lives at the bottom of the notebook in the **Reflection** section, but the headline observations:

- **Chain-of-Thought was the most reliable winner.** Reading the trace, you can see the model doing the right kind of bookkeeping. The Baseline solved the easy openers but lost the plot on harder words.
- **Self-Reflection didn't actually self-correct.** I found cases where the loop went `built → guilt → built` over three "reflection" attempts and then submitted `built` anyway. The proposer kept ignoring the critic.
- **The model would contradict its own analysis.** The critic would correctly list "U is GREY" in step 3, then conclude in step 5 that a word containing U is fine. Long step-by-step reasoning produces an *illusion* of correctness that's hard to verify by skimming.
- **Self-Consistency helped late-game (when the answer space was tight) and was useless early-game** (three random openers don't aggregate into anything meaningful).

The deeper takeaway, written up properly in the notebook: **Wordle isn't really a reasoning problem — it's a constraint-satisfaction problem dressed up as one.** The LLM is the wrong tool for the bookkeeping part (which is mechanical) and the right tool for the vocabulary-recall part (which is its strength). The cleanest fix would be to replace the LLM critic with a deterministic `is_consistent()` check, and keep the model only for proposing candidates and picking among consistent ones.

## What I'd Do Differently

Outlined in detail in the notebook reflection, but tl;dr:

1. Replace the LLM critic with a deterministic constraint checker that uses `score_guess` directly. Cheaper, faster, and actually correct.
2. Detect within-loop proposal repeats in Self-Reflection so the `built → guilt → built` trap can't happen.
3. Tighten `_parse_critique`: take the *last* `VERDICT:` line (not the first match), and fail closed (treat unparseable critiques as INVALID), not open.
4. Fix the duplicate-letter handling in `count_constraints_satisfied` for Self-Consistency tie-breaks.

## Some Bugs I Found Along the Way

A couple of debugging stories worth keeping for the next person who builds on this:

- **The `reasoning` flag was silently being ignored.** I'd been passing `reasoning={"reasoning": {"enabled": True}}` (a doubly-nested dict) to `ChatOpenRouter`. OpenRouter expects `reasoning={"enabled": True}` flat, and silently dropped my malformed key. So all the "reasoning ON" runs were actually running with reasoning OFF the whole time. Verified by `curl`-ing the OpenRouter API directly and checking `usage.completion_tokens_details.reasoning_tokens`.
- **One provider rate-limiting causes 429s and 504s** even when you have credits. Fix: pass `openrouter_provider={"sort": "throughput", "allow_fallbacks": True}` so OpenRouter picks the least-loaded provider per call.
- **Picking the right model matters more than picking the right prompting strategy.** I started on a dense 31B model and was getting timeouts. Switched to `google/gemma-4-26b-a4b-it` (MoE with only 4B active params) and the throughput improved enough that the experiments became practical.

## Tech Stack

- **LangGraph** — explicit state machine for the game loop. Worth using over a hand-rolled while-loop because the conditional edges (`won` / `lost` / `continue`) make the control flow visually obvious and easy to extend.
- **LangChain** — prompt templates and structured-output parsing (Pydantic-backed JSON-schema mode for the agents that need a strict format).
- **OpenRouter** — single API key, pick-your-model. `langchain_openrouter` exposes a drop-in `ChatOpenRouter` with reasoning support and provider routing.
- **Google Gemma 4 26B-A4B-IT** — MoE, 4B active parameters per token. Fast enough to make the multi-call agents practical.

## Repository Layout

```
.
├── README.md                      # this file
├── genai-2-assignment.ipynb       # the notebook — agents, experiments, reflection
├── genai-assignment-2.pdf         # original assignment brief
├── .env.example                   # template for required API keys
└── .gitignore
```

---

Built as part of a Generative AI course assignment. The full reflection — including what surprised me and what I'd build next — is at the bottom of the notebook.
