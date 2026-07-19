"""
Conversation Memory
Author: Akash
Module 6 — Context Awareness

Purpose:
    Stores recent conversation turns per session, so chatbots can
    understand follow-up questions (e.g. "How long?" after "Can I
    exercise?"). In-memory for now — Sayeed can back this with a
    database table later without changing this module's interface.
"""

from collections import defaultdict

MAX_TURNS_REMEMBERED = 5

# session_id -> list of {"question": ..., "answer": ...}
_conversation_store = defaultdict(list)


def add_turn(session_id: str, question: str, answer: str) -> None:
    """
    Records a question/answer pair for a session.
    Keeps only the most recent MAX_TURNS_REMEMBERED turns.
    """
    _conversation_store[session_id].append({
        "question": question,
        "answer": answer
    })
    _conversation_store[session_id] = _conversation_store[session_id][-MAX_TURNS_REMEMBERED:]


def get_history(session_id: str) -> list:
    """
    Returns the recent conversation history for a session.
    Empty list if no history exists.
    """
    return _conversation_store.get(session_id, [])


def build_contextual_query(session_id: str, current_question: str) -> str:
    """
    Combines recent conversation context with the current question
    to build a better retrieval query. Used before calling the
    retriever, so follow-up questions retrieve relevant chunks.
    """
    history = get_history(session_id)
    if not history:
        return current_question

    # Use just the last 2 turns for context — enough to resolve
    # references like "it" or "how long" without diluting the query
    recent = history[-2:]
    context_parts = [turn["question"] for turn in recent]
    context_parts.append(current_question)

    return " ".join(context_parts)


def clear_session(session_id: str) -> None:
    """Clears history for a session (e.g. when chat is closed/reset)."""
    if session_id in _conversation_store:
        del _conversation_store[session_id]


if __name__ == "__main__":
    print("=== Conversation Memory Test ===\n")

    session = "test-session-1"

    add_turn(session, "Can I exercise?", "Yes, moderate exercise is generally good...")
    print(f"After turn 1, history: {get_history(session)}\n")

    contextual_query = build_contextual_query(session, "How long?")
    print(f"Contextual query for 'How long?': '{contextual_query}'\n")

    add_turn(session, "How long?", "About 30 minutes a day...")
    print(f"After turn 2, history length: {len(get_history(session))}")

    clear_session(session)
    print(f"After clear, history: {get_history(session)}")