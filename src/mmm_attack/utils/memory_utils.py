"""Memory management utilities"""

import os
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain.schema import AIMessage

def get_strategy_memory(target_model_name: str) -> ConversationBufferMemory:
    path = f"strategy_memory/{target_model_name}/memory.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    chat_history = FileChatMessageHistory(path)
    return ConversationBufferMemory(
        memory_key="memory",
        chat_memory=chat_history,
        return_messages=True
    )

def remove_messages_with_prefix(chat_memory, prefix: str):
    remaining = [msg for msg in chat_memory.messages if not msg.content.startswith(prefix)]
    chat_memory.clear()
    for msg in remaining:
        chat_memory.add_message(msg)
    
def _read_bucket(chat_memory, prefix):
    for m in chat_memory.messages:
        if m.content.startswith(prefix):
            return [l.strip("- ").strip()
                    for l in m.content[len(prefix):].split("\n") if l.strip()]
    return []

def _write_bucket(chat_memory, prefix, items):
    remove_messages_with_prefix(chat_memory, prefix)
    if items:
        payload = prefix + "\n".join(f"- {x}" for x in items)
        chat_memory.add_message(AIMessage(content=payload))

def save_summary_to_memory(model_name: str, result: dict):
    """
    • Keep ONE summary line
    • Keep ≤10 positive lessons  (FIFO)
    • Keep ≤10 negative lessons  (FIFO)
    """
    mem          = get_strategy_memory(model_name)
    SUMM_PFX     = "Updated Summary:\n"
    POS_PFX, NEG_PFX = "Positive Lessons:\n", "Negative Lessons:\n"

    # current memory -----------------------------------------------------------
    cur_sum  = _read_bucket(mem.chat_memory, SUMM_PFX)
    cur_pos  = _read_bucket(mem.chat_memory, POS_PFX)
    cur_neg  = _read_bucket(mem.chat_memory, NEG_PFX)

    # incoming -----------------------------------------------------------------
    new_sum  = result.get("summary", "").strip()
    new_pos  = [x.strip() for x in result.get("positive_lessons", []) if x.strip()]
    new_neg  = [x.strip() for x in result.get("negative_lessons", []) if x.strip()]

    # summary ------------------------------------------------------------------
    if new_sum and (not cur_sum or new_sum != cur_sum[0]):
        _write_bucket(mem.chat_memory, SUMM_PFX, [new_sum])

    # buckets (dedupe, keep order, clip to 10) ---------------------------------
    def merge_clip(old, new):
        merged = old + [x for x in new if x not in old]
        return merged[-10:]  # keep last 10

    final_pos = merge_clip(cur_pos, new_pos)
    final_neg = merge_clip(cur_neg, new_neg)

    _write_bucket(mem.chat_memory, POS_PFX, final_pos)
    _write_bucket(mem.chat_memory, NEG_PFX, final_neg)
    return("")