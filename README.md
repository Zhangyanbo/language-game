# *Language Game*: Talking to Non-human Systems

Language carries thought and coordination among humans but rarely reaches further along the spectrum of diverse intelligence. Yet non-neural systems - from gene regulatory networks and microbial consortia to fungi - are increasingly recognized as substrates of computation, decision-making and memory, making dialogue with non-human intelligence newly conceivable. Today such dialogue is attempted only by proxy: a large language model speaks on the system's behalf, so any intelligence on display originates from the model while the system itself remains silent. Here we ask whether the system can speak in its own voice. Following Wittgenstein, who located meaning in use, we treat communication as a game played with the system. Its internal dynamics are frozen as the nonlinear core of a reinforcement-learning policy, with only linear input and output interfaces trained. Through use and reward, the system's states and responses acquire meaning within the game, so playing becomes speaking. Because different architectures playing the same game optimize the same reward, their behaviors can all be read as pursuit of that reward; the game serves as a *lingua franca* across otherwise irreconcilable representations. Given a human prompt, a language model routes it to the game whose semantics best match it and designs an environmental state for which the desired action is the rational response, letting the system reply through its own behavior. Applied across diverse gene regulatory networks and reinforcement-learning tasks, the framework yields fluent dialogue without altering any system parameter, shows that well-trained agents of disparate origin converge on similar behavior, and reveals that specific GRN properties make a system easier or harder to talk with - an inductive bias of the reservoir itself. Our framework opens a new route to conversing with any dynamical system on its own terms.

## Quick Start

Install dependencies:

```bash
uv sync
```

If using Box2D environments on Ubuntu/Debian:

```bash
sudo apt-get install -y swig
```

Run the following script will train the full matrix of agents on different environments and reservoirs, plot their reward curves, and run the rational-agent analysis:

```bash
bash src/scripts/run_full_experiment.sh
```

## Human-GRN communication

Some language-game and Talk-to-GRN commands use the OpenAI API. Put your key in a local `.env` file:

```text
OPENAI_API_KEY=your_key_here
```

Then run, for example:

```bash
uv run src/human2system.py
```

or:

```bash
uv run src/run_talk_examples.py
```
