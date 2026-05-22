"""Interactive chat: conversation state and a console REPL.

The model is a plain next-token predictor; "chat" is just a convention for how
the conversation is laid out in the token stream::

    <bos> <|system|> system text
          <|user|> first user message
          <|assistant|> first model reply <eos>
          <|user|> second user message
          <|assistant|>   <- the model continues from here

:class:`ChatSession` maintains the running history, re-renders it into this
layout before every turn, and trims the oldest turns when it would overflow the
context window.
"""

from __future__ import annotations

from typing import Callable, Iterator, List, Optional, Tuple

from ..sampling import SamplingConfig
from ..tokenizer.bpe import ASSISTANT, SYSTEM, USER, StreamDecoder
from .engine import InferenceEngine


class ChatSession:
    """Stateful multi-turn conversation on top of an :class:`InferenceEngine`."""

    def __init__(
        self,
        engine: InferenceEngine,
        *,
        system_prompt: Optional[str] = None,
        sampling: Optional[SamplingConfig] = None,
        max_new_tokens: int = 200,
    ) -> None:
        self.engine = engine
        tokenizer = engine.tokenizer
        for name in (SYSTEM, USER, ASSISTANT):
            if name not in tokenizer.special_tokens:
                raise ValueError(
                    f"tokenizer is missing the chat special token {name!r}; "
                    "retrain it with the default special tokens"
                )
        self.system_prompt = system_prompt
        self.sampling = sampling or SamplingConfig()
        self.max_new_tokens = max_new_tokens
        self.history: List[Tuple[str, str]] = []

    def reset(self) -> None:
        """Clear the conversation history."""
        self.history.clear()

    def _encode_turn(self, role_token: str, text: str) -> List[int]:
        tokenizer = self.engine.tokenizer
        return [tokenizer.special_id(role_token)] + tokenizer.encode_ordinary(text)

    def _build_prompt(self) -> List[int]:
        """Render the full conversation into a token id sequence."""
        tokenizer = self.engine.tokenizer
        ids: List[int] = []
        if tokenizer.bos_id is not None:
            ids.append(tokenizer.bos_id)
        if self.system_prompt:
            ids += self._encode_turn(SYSTEM, self.system_prompt)
        for role, text in self.history:
            if role == "user":
                ids += self._encode_turn(USER, text)
            else:
                ids += self._encode_turn(ASSISTANT, text)
                if tokenizer.eos_id is not None:
                    ids.append(tokenizer.eos_id)
        # Open an assistant turn for the model to complete.
        ids.append(tokenizer.special_id(ASSISTANT))
        return ids

    def _build_trimmed_prompt(self) -> List[int]:
        """Build the prompt, dropping the oldest turns if it would overflow."""
        budget = max(self.engine.max_context - self.max_new_tokens - 1, 16)
        ids = self._build_prompt()
        while len(ids) > budget and len(self.history) > 1:
            self.history.pop(0)
            ids = self._build_prompt()
        limit = self.engine.max_context - 1
        if len(ids) > limit:
            ids = ids[-limit:]
        return ids

    def stream(self, user_text: str) -> Iterator[str]:
        """Append a user message and yield the reply fragment by fragment.

        Both the user message and the finished reply are recorded in
        :attr:`history`.
        """
        tokenizer = self.engine.tokenizer
        self.history.append(("user", user_text))
        prompt_ids = self._build_trimmed_prompt()

        # Stop the reply at <eos> or as soon as the model opens a new user turn.
        stop_ids: List[int] = []
        if tokenizer.eos_id is not None:
            stop_ids.append(tokenizer.eos_id)
        stop_ids.append(tokenizer.special_id(USER))

        decoder = StreamDecoder(tokenizer)
        pieces: List[str] = []
        for token_id in self.engine.generate_stream(
            prompt_ids,
            max_new_tokens=self.max_new_tokens,
            sampling=self.sampling,
            stop_ids=stop_ids,
        ):
            fragment = decoder.step(token_id)
            if fragment:
                pieces.append(fragment)
                yield fragment
        tail = decoder.flush()
        if tail:
            pieces.append(tail)
            yield tail
        self.history.append(("assistant", "".join(pieces).strip()))

    def send(
        self, user_text: str, on_token: Optional[Callable[[str], None]] = None
    ) -> str:
        """Append a user message, generate a reply, and return it.

        ``on_token`` (if given) receives each fragment as it is produced.
        """
        pieces: List[str] = []
        for fragment in self.stream(user_text):
            pieces.append(fragment)
            if on_token is not None:
                on_token(fragment)
        return "".join(pieces).strip()


def run_chat(
    engine: InferenceEngine,
    *,
    system_prompt: Optional[str] = None,
    sampling: Optional[SamplingConfig] = None,
    max_new_tokens: int = 200,
) -> None:
    """Run a blocking interactive chat REPL on the console.

    Commands: ``/reset`` clears the history, ``/exit`` (or Ctrl-D/Ctrl-C) quits.
    """
    session = ChatSession(
        engine,
        system_prompt=system_prompt,
        sampling=sampling,
        max_new_tokens=max_new_tokens,
    )
    print("LocalNeuro interactive chat.  Commands: /reset  /exit")
    print("-" * 60)
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            session.reset()
            print("(conversation cleared)")
            continue
        print("bot> ", end="", flush=True)
        session.send(user, on_token=lambda fragment: print(fragment, end="", flush=True))
        print()
    print("Goodbye.")
