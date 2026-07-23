---
title: Before you arrive
---

# Before you arrive

Almost every problem a participant hits at a bootcamp is one we could have solved
a fortnight earlier over email. This page is the short list of things worth doing
in advance, roughly in the order you should do them.

!!! danger "The one rule"
    **Do not install anything for the first time at the venue.**

    PyTorch and MuJoCo are large downloads. A room full of people fetching them
    simultaneously over conference Wi-Fi is slower than you think, and it costs
    the whole group the first session. Install at home, on a connection you
    trust.

---

## A sensible timeline

=== "Two weeks before"

    - [ ] Check you can install software on the laptop you plan to bring
          (see [laptop reality check](#laptop-reality-check) below).
    - [ ] Follow the [installation guide](setup/installation.md) all the way to
          the smoke test.
    - [ ] If anything fails, report it now. Two weeks is enough time to fix
          anything; two minutes is not.

=== "One week before"

    - [ ] Run the [worked example](basics/worked-example.md) end to end. Train an
          agent, look at the TensorBoard curve, watch the video.
    - [ ] Skim the [Gymnasium](basics/gymnasium.md) and
          [Stable-Baselines3](basics/stable-baselines3.md) pages so the API is
          familiar rather than new.
    - [ ] Brush up on whatever is on your [self-assessment](#self-assessment)
          list.

=== "The night before"

    - [ ] `conda activate rlbootcamp && python scripts/smoke_test.py` one more
          time. Environments rot — an OS update or an unrelated `pip install`
          can break things.
    - [ ] Charge your laptop, pack the charger and a plug adapter.
    - [ ] Download anything you want offline. Assume the Wi-Fi will disappoint
          you.

---

## Laptop reality check {#laptop-reality-check}

**You do not need a powerful machine.** Every exercise is sized to run on a
laptop CPU. What you actually need:

| Requirement | Detail |
|---|---|
| **Disk** | ~5 GB free. PyTorch alone is a couple of GB. |
| **RAM** | 8 GB is enough; 16 GB is comfortable. |
| **CPU** | Any machine from the last ~6 years. |
| **GPU** | **Not required.** Small MLP policies train *faster* on CPU — the data transfer costs more than the matrix multiplies save. |
| **OS** | Linux, macOS (Intel or Apple Silicon), or Windows. All three are supported and tested. |
| **Admin rights** | You must be able to install software and create files in your home directory. |

!!! warning "Corporate and university-managed laptops"
    This is the single most common blocker, and it is invisible until you try:

    - **Locked-down installers.** Some managed machines refuse unsigned
      installers outright. Conda installs entirely inside your home directory
      and usually survives this — but test it early.
    - **Proxies and VPNs.** Corporate proxies frequently break `conda` and `pip`
      with TLS errors. See the
      [proxy fixes](setup/troubleshooting.md#proxies-and-corporate-networks).
    - **Antivirus.** Real-time scanners can quarantine files mid-install,
      producing baffling half-broken environments.
    - **Disk quotas.** A 2 GB home-directory quota will not fit PyTorch.

    If your work laptop fights you, **bring a personal machine instead**. If you
    have neither, tell the organisers *before* the event — do not turn up hoping
    it will work out.

---

## What you should already know

The tutorial track assumes working Python. It does **not** assume you have done
RL before — but the more of the list below you are comfortable with, the more you
will get out of the advanced material.

### Self-assessment {#self-assessment}

Tick honestly. Anything you cannot tick is a good use of an evening beforehand.

**Python — you will struggle without these:**

- [ ] Writing and running a script from a terminal, and passing it arguments.
- [ ] Virtual environments: what they are and why the wrong one breaks imports.
- [ ] NumPy arrays: indexing, shapes, broadcasting, `dtype`.
- [ ] Reading a traceback and finding the line that actually failed.
- [ ] Classes: `__init__`, methods, inheritance, subclassing someone else's class.

**RL concepts — helpful, and covered from the beginning if not:**

- [ ] Agent, environment, state, action, reward, episode.
- [ ] The idea of a *policy* and of a *value function*.
- [ ] Discounting, and why a discount factor exists at all.
- [ ] Exploration versus exploitation.
- [ ] Roughly what "on-policy" and "off-policy" mean.

**Nice to have:**

- [ ] PyTorch basics: tensors, `nn.Module`, an optimiser step.
- [ ] Reading a learning curve and distinguishing noise from a trend.
- [ ] Git: clone, pull, branch.

### If you want to prepare properly

Ordered by return on time invested:

1. **Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.),
   chapters 1–6.** The canonical text, and
   [free online](http://incompleteideas.net/book/the-book-2nd.html). Chapters 3
   (MDPs), 4 (dynamic programming) and 6 (temporal-difference learning) are the
   backbone of any bootcamp's first session.
2. **The [Gymnasium documentation](https://gymnasium.farama.org/) "Basic Usage"
   page.** Twenty minutes, and it is the API you will type all day.
3. **The [Stable-Baselines3 "Getting Started"](https://stable-baselines3.readthedocs.io/)
   guide.** Another twenty minutes.
4. **[Spinning Up in Deep RL](https://spinningup.openai.com/)** (OpenAI) — the
   best free bridge from "I know what a policy is" to "I know what PPO does".
5. **The [worked example on this site](basics/worked-example.md).** Concrete,
   runnable, and about half an hour of wall-clock time.

Do not try to do all five. One and five is a good evening.

---

## Set your expectations about RL

Newcomers are often surprised by how RL *feels* in practice. Knowing this in
advance saves a lot of frustration:

- **It is slow.** Training the small example on this site takes several minutes
  for a task you could hand-code in twenty lines. Real tasks take hours. This is
  normal and is not your machine being broken.
- **It is noisy.** Two runs that differ only in random seed can produce visibly
  different curves. A single run tells you very little. Whenever you can afford
  it, run several seeds — and be suspicious of anyone (including yourself) who
  reports one.
- **Reward curves lie.** A rising `ep_rew_mean` means the agent is getting better
  at maximising *your reward function*, which is not the same as getting better
  at *your task*. Agents are excellent at finding the gap between the two.
- **Most failures are silent.** RL code rarely crashes; it just quietly learns
  nothing. Baselines are how you notice.
- **The baseline usually deserves to win.** A well-chosen heuristic beats a
  half-trained agent surprisingly often. That is a legitimate result and worth
  reporting, not a failure to hide.

!!! quote "The question to keep asking"
    *"Is this actually better than the obvious thing?"*

    Sound RL practice is mostly the discipline of answering that honestly. If a
    simple rule beats your agent, either the task does not need RL, or your
    reward and observation design are hiding the signal. Both are useful findings.

---

## Practicalities

- **Bring:** laptop, charger, and a Type-F/Type-C plug adapter if you are
  travelling to continental Europe.
- **Headphones** are genuinely useful — some exercises render video, and a room
  of unmuted laptops is unpleasant.
- **A mouse** helps if you dislike trackpads; there is a fair amount of plotting.
- **Pair up.** Sessions are more productive in pairs, and debugging someone
  else's environment is the fastest way to understand your own.
- **Ask early.** Helpers circulate throughout. Ten minutes stuck is normal;
  forty-five minutes stuck and silent is a waste of your day.

---

## Getting help before the event

If the [troubleshooting page](setup/troubleshooting.md) does not solve it, open
an issue on this repository — or email the organisers — with:

1. Your **OS and version**, and whether it is an Apple Silicon Mac.
2. Whether you used **Conda or a plain virtualenv**.
3. The **entire output** of `python scripts/smoke_test.py`, not just the last
   line.
4. The output of `conda run -n rlbootcamp python -c "import sys; print(sys.executable)"`.

That is almost always enough to diagnose it remotely.

---

Next: [Installation →](setup/installation.md)
