"""Terminal knowledge-base chatbot powered by the Google Gemini API."""

import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from knowledge_base import KnowledgeBase

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODELS = [
    name.strip()
    for name in os.getenv(
        "GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-3.6-flash"
    ).split(",")
    if name.strip()
]
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a knowledge-base assistant. Answer questions using ONLY the "
    "CONTEXT provided in the user message. Never rely on outside knowledge. "
    "If the context does not contain the answer, say that the knowledge base "
    "does not cover it. Cite the sources you use as [1], [2], and so on. "
    "Be clear and concise.",
)


def get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Error: GEMINI_API_KEY is not set. Add it to the .env file.")
    return api_key


def models_to_try() -> list[str]:
    seen: set[str] = set()
    models: list[str] = []
    for name in [MODEL, *FALLBACK_MODELS]:
        if name not in seen:
            seen.add(name)
            models.append(name)
    return models


def is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    return code in {429, 500, 503} or "UNAVAILABLE" in str(exc)


def is_quota_error(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(exc)


def make_contents(
    history: list[types.Content], prompt: str
) -> list[types.Content]:
    return [
        *history,
        types.Content(role="user", parts=[types.Part(text=prompt)]),
    ]


def stream_reply(client: genai.Client, history, prompt: str, retries: int = 2):
    """Stream a reply, retrying and falling back to other models if needed."""
    contents = make_contents(history, prompt)
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
    last_error: Exception | None = None

    for model in models_to_try():
        for attempt in range(retries):
            emitted = False
            try:
                for chunk in client.models.generate_content_stream(
                    model=model, contents=contents, config=config
                ):
                    if chunk.text:
                        emitted = True
                        yield chunk.text
                return
            except Exception as exc:
                last_error = exc
                if emitted or not is_retryable(exc) or attempt == retries - 1:
                    break
                time.sleep(2**attempt)

    if last_error is not None:
        raise last_error


def build_prompt(question: str, matches) -> str:
    context = "\n\n".join(
        f"[{index}] Source: {match.chunk.source}\n{match.chunk.text}"
        for index, match in enumerate(matches, start=1)
    )
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer the question using only the context above. If the answer is "
        "not in the context, say the knowledge base does not cover it. "
        "Cite sources as [1], [2], and so on."
    )


def load_knowledge_base(client: genai.Client) -> KnowledgeBase:
    kb = KnowledgeBase(client)
    print("Loading knowledge base...", flush=True)
    kb.refresh()
    return kb


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    client = genai.Client(api_key=get_api_key())
    try:
        kb = load_knowledge_base(client)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"Error: {exc}")
    except Exception as exc:
        sys.exit(f"Error loading the knowledge base: {exc}")

    history: list[types.Content] = []

    print(f"Knowledge Base Chatbot ({MODEL})")
    print(f"Loaded {len(kb.documents)} document(s), {len(kb.chunks)} chunk(s).")
    print("Commands: /sources, /reload, /reset, /exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        command = user_input.lower()
        if command in {"/exit", "/quit"}:
            print("Goodbye!")
            break
        if command == "/reset":
            history.clear()
            print("Conversation cleared.\n")
            continue
        if command == "/sources":
            print("Knowledge base files:")
            for source in kb.sources():
                print(f"  - {source}")
            print()
            continue
        if command == "/reload":
            try:
                kb.refresh()
                print(
                    f"Reloaded {len(kb.documents)} document(s), "
                    f"{len(kb.chunks)} chunk(s).\n"
                )
            except Exception as exc:
                print(f"Could not reload the knowledge base: {exc}\n")
            continue

        try:
            matches = kb.retrieve(user_input)
        except Exception as exc:
            if is_quota_error(exc):
                print("Embedding quota exceeded. Please try again later.\n")
            elif is_retryable(exc):
                print("The embedding service is busy. Please try again.\n")
            else:
                print(f"Error: {exc}\n")
            continue

        if not matches:
            print(
                "Bot: I could not find anything about that in the knowledge "
                "base. Try rephrasing, or add a document to knowledge/.\n"
            )
            continue

        prompt = build_prompt(user_input, matches)
        print("Bot: ", end="", flush=True)
        answer_parts: list[str] = []
        try:
            for text in stream_reply(client, history, prompt):
                answer_parts.append(text)
                print(text, end="", flush=True)
        except Exception as exc:
            if is_quota_error(exc):
                print(
                    "\nAPI quota exceeded. Wait for the free-tier limit to "
                    "reset, or set another GEMINI_MODEL in .env."
                )
            elif is_retryable(exc):
                print("\nThe model is busy right now. Please try again in a moment.")
            else:
                print(f"\nError: {exc}")
        print("\n")

        answer = "".join(answer_parts).strip()
        if answer:
            history.append(
                types.Content(role="user", parts=[types.Part(text=prompt)])
            )
            history.append(
                types.Content(role="model", parts=[types.Part(text=answer)])
            )

        cited = ", ".join(
            f"[{index}] {match.chunk.source}"
            for index, match in enumerate(matches, start=1)
        )
        print(f"Sources: {cited}\n")


if __name__ == "__main__":
    main()
