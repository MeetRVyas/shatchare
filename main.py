import asyncio
import json
import logging
import random
import string
from collections import defaultdict
from typing import Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("llm-rooms")

app = FastAPI(title="LLM Chat Rooms")

# ─── Constants ────────────────────────────────────────────────────────────────

GEMINI_MODELS = {
    "gemini-2.0-flash-thinking-exp": "Gemini 2.0 Flash Thinking",
    "gemini-2.0-flash-lite":         "Gemini 2.0 Flash Lite",
}

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# ─── In-memory state ──────────────────────────────────────────────────────────

rooms: dict[str, dict] = {}
room_connections: dict[str, set[WebSocket]] = defaultdict(set)

# Cap concurrent Gemini calls across all rooms
LLM_SEMAPHORE = asyncio.Semaphore(10)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_code(length: int = 6) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def default_room() -> dict:
    return {
        "config": {
            "model": "gemini-2.0-flash-thinking-exp",
            "system_prompt": "You are a helpful assistant.",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        "api_keys": [],
        "messages": [],
    }


def mask_key(key: str) -> str:
    if len(key) <= 8:
        return "****"
    return key[:4] + "****" + key[-4:]


def room_state_payload(room_id: str) -> dict:
    room = rooms[room_id]
    return {
        "type": "room_state",
        "room_id": room_id,
        "config": room["config"],
        "api_keys": [mask_key(k) for k in room["api_keys"]],
        "key_count": len(room["api_keys"]),
        "messages": room["messages"],
    }


async def broadcast(room_id: str, payload: dict, exclude: Optional[WebSocket] = None):
    dead = set()
    for ws in list(room_connections[room_id]):
        if ws is exclude:
            continue
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            dead.add(ws)
    for ws in dead:
        room_connections[room_id].discard(ws)


# ─── Gemini API ───────────────────────────────────────────────────────────────

RATE_LIMIT_PHRASES = ["quota", "rate", "limit", "exhausted", "resource_exhausted"]


def is_rate_limit(status: int, body: dict) -> bool:
    if status == 429:
        return True
    txt = json.dumps(body).lower()
    return any(p in txt for p in RATE_LIMIT_PHRASES)


class RateLimitError(Exception):
    pass


class GeminiError(Exception):
    pass


async def call_with_key(
    api_key: str,
    model: str,
    system_prompt: str,
    history: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    url = f"{GEMINI_BASE}/{model}:generateContent?key={api_key}"

    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload)

    body = resp.json()

    if resp.status_code != 200:
        if is_rate_limit(resp.status_code, body):
            raise RateLimitError(f"Rate limit on key ...{api_key[-4:]}")
        msg = body.get("error", {}).get("message", str(body))
        raise GeminiError(f"Gemini {resp.status_code}: {msg}")

    try:
        return body["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise GeminiError(f"Unexpected response: {body}") from e


async def call_gemini(room: dict, history: list[dict], user_message: str) -> tuple[str, list[str]]:
    """
    Rotate through available keys. Remove keys that hit rate limits.
    Returns (reply_text, list_of_removed_keys).
    """
    cfg = room["config"]
    keys = list(room["api_keys"])

    if not keys:
        raise GeminiError("No API keys configured. Add at least one key in Settings.")

    full_history = history + [{"role": "user", "content": user_message}]
    removed: list[str] = []

    for key in keys:
        try:
            async with LLM_SEMAPHORE:
                result = await call_with_key(
                    api_key=key,
                    model=cfg["model"],
                    system_prompt=cfg["system_prompt"],
                    history=history,
                    temperature=float(cfg["temperature"]),
                    max_tokens=int(cfg["max_tokens"]),
                )
            return result, removed
        except RateLimitError:
            log.warning(f"Rate limit — removing key ...{key[-4:]}")
            removed.append(key)
            if key in room["api_keys"]:
                room["api_keys"].remove(key)

    raise GeminiError(
        "All API keys have hit their rate limits and were removed. Please add new keys."
    )


# ─── REST ─────────────────────────────────────────────────────────────────────

@app.post("/rooms")
async def create_room():
    room_id = make_code()
    while room_id in rooms:
        room_id = make_code()
    rooms[room_id] = default_room()
    log.info(f"Room created: {room_id}")
    return {"room_id": room_id}


@app.get("/rooms/{room_id}/exists")
async def room_exists(room_id: str):
    return {"exists": room_id.upper() in rooms}


# ─── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(ws: WebSocket, room_id: str):
    room_id = room_id.upper()
    if room_id not in rooms:
        await ws.accept()
        await ws.send_text(json.dumps({"type": "error", "message": "Room not found."}))
        await ws.close()
        return

    await ws.accept()
    room_connections[room_id].add(ws)
    room = rooms[room_id]

    await ws.send_text(json.dumps(room_state_payload(room_id)))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            mtype = msg.get("type")

            if mtype == "add_key":
                key = msg.get("key", "").strip()
                if key and key not in room["api_keys"]:
                    room["api_keys"].append(key)
                    await broadcast(room_id, {
                        "type": "keys_updated",
                        "api_keys": [mask_key(k) for k in room["api_keys"]],
                        "key_count": len(room["api_keys"]),
                        "notice": f"Key {mask_key(key)} added.",
                    })

            elif mtype == "remove_key":
                idx = msg.get("index")
                if idx is not None and 0 <= int(idx) < len(room["api_keys"]):
                    removed = room["api_keys"].pop(int(idx))
                    await broadcast(room_id, {
                        "type": "keys_updated",
                        "api_keys": [mask_key(k) for k in room["api_keys"]],
                        "key_count": len(room["api_keys"]),
                        "notice": f"Key {mask_key(removed)} removed.",
                    })

            elif mtype == "update_config":
                updates = msg.get("config", {})
                allowed = {"model", "system_prompt", "temperature", "max_tokens"}
                for k, v in updates.items():
                    if k in allowed:
                        room["config"][k] = v
                await broadcast(room_id, {
                    "type": "config_updated",
                    "config": room["config"],
                })

            elif mtype == "chat":
                user_text = msg.get("content", "").strip()
                username = msg.get("username", "Anonymous")
                if not user_text:
                    continue

                user_msg = {
                    "role": "user",
                    "content": user_text,
                    "username": username,
                    "ts": asyncio.get_event_loop().time(),
                }
                room["messages"].append(user_msg)
                await broadcast(room_id, {"type": "message", "message": user_msg})
                await broadcast(room_id, {"type": "typing", "active": True})

                asyncio.create_task(
                    handle_llm(room_id, room, list(room["messages"]))
                )

    except WebSocketDisconnect:
        pass
    finally:
        room_connections[room_id].discard(ws)


async def handle_llm(room_id: str, room: dict, history: list[dict]):
    user_message = history[-1]["content"]
    prior = history[:-1]

    try:
        reply_text, removed_keys = await call_gemini(room, prior, user_message)

        ai_msg = {
            "role": "assistant",
            "content": reply_text,
            "username": "Assistant",
            "ts": asyncio.get_event_loop().time(),
        }
        room["messages"].append(ai_msg)
        await broadcast(room_id, {"type": "typing", "active": False})
        await broadcast(room_id, {"type": "message", "message": ai_msg})

        if removed_keys:
            await broadcast(room_id, {
                "type": "keys_updated",
                "api_keys": [mask_key(k) for k in room["api_keys"]],
                "key_count": len(room["api_keys"]),
                "notice": f"⚠ {len(removed_keys)} key(s) hit rate limits and were removed automatically.",
            })

    except GeminiError as e:
        await broadcast(room_id, {"type": "typing", "active": False})
        await broadcast(room_id, {"type": "error", "message": str(e)})
    except Exception as e:
        log.exception("Unexpected error in handle_llm")
        await broadcast(room_id, {"type": "typing", "active": False})
        await broadcast(room_id, {"type": "error", "message": f"Internal error: {e}"})


# ─── Frontend ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", encoding="utf-8") as f:
        return f.read()
