"""
Minimal Groq client wrapper - FIXED VERSION with proper typing
"""
import os
import json
import requests
import time
from typing import List, Dict, Any, Iterable, Optional, cast
from ..config import Config

# Try official package first
try:
    from groq import Groq
    _HAS_GROQ = False  # Disable due to proxies argument issue; use HTTP fallback
except Exception:
    Groq = None
    _HAS_GROQ = False

GROQ_API = "https://api.groq.com/openai/v1/chat/completions"

def _validate_api_key() -> str:
    api_key = Config.GROQ_API_KEY
    if not api_key or not isinstance(api_key, str):
        raise RuntimeError("GROQ_API_KEY not set in environment")
    return api_key

def _normalize_messages(messages: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    msgs = []
    for m in messages:
        if isinstance(m, dict) and 'role' in m and 'content' in m:
            msgs.append({"role": str(m['role']), "content": str(m['content'])})
        elif isinstance(m, str):
            msgs.append({"role": "user", "content": m})
        else:
            msgs.append({"role": str(m.get('role','user')), "content": str(m.get('content',''))})
    return msgs

def _get_valid_model(model: Optional[str]) -> str:
    if model is None:
        model = Config.GROQ_MODEL
    if model is None:
        model = "llama-3.3-70b-versatile"
    model = str(model).strip()
    # Map deprecated models
    if model in ["llama3-70b-8192", "llama3-8b-8192"]:
        model = "llama-3.3-70b-versatile"
    return model

def groq_chat(
    messages: Iterable[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 1.2,
    max_tokens: int = 512
) -> Dict[str, Any]:
    api_key = _validate_api_key()
    model = _get_valid_model(model)
    msgs = _normalize_messages(messages)
    
    if not msgs:
        raise ValueError("Messages list cannot be empty")

    # FIX: Proper typing for Groq client
    if _HAS_GROQ and Groq is not None:
        try:
            # Initialize Groq client with only supported parameters
            client = Groq(api_key=api_key, timeout=60.0)
            
            # Cast to Any to bypass strict type checking (Groq's typing is flexible)
            messages_param: Any = msgs
            
            resp = client.chat.completions.create(
                model=model,
                messages=messages_param,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Safe attribute access with fallbacks
            choices = resp.choices if resp.choices else []
            first_choice = choices[0] if choices else None
            
            if not first_choice:
                raise RuntimeError("No choices returned from Groq API")
            
            message_content = first_choice.message.content if first_choice.message else "No content"
            finish_reason = first_choice.finish_reason if hasattr(first_choice, 'finish_reason') else "unknown"
            
            # Safe usage extraction
            usage = resp.usage if hasattr(resp, 'usage') and resp.usage else None
            
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": message_content
                    },
                    "finish_reason": finish_reason
                }],
                "model": resp.model if hasattr(resp, 'model') else model,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0
                }
            }

        except Exception as e:
            print(f"Groq client error (falling back to HTTP): {e}")
            # Continue to HTTP fallback

    # HTTP Fallback with retry
    payload = {
        "model": model,
        "messages": msgs,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for attempt in range(3):
        try:
            r = requests.post(GROQ_API, headers=headers, json=payload, timeout=60)
            
            if r.status_code != 200:
                error_detail = r.text
                try:
                    error_json = r.json()
                    error_detail = error_json.get('error', {}).get('message', error_detail)
                except:
                    pass
                raise RuntimeError(f"Groq API error (HTTP {r.status_code}): {error_detail}")
            
            return r.json()
            
        except RuntimeError as e:
            if attempt < 2:
                print(f"Retry {attempt + 1}/3 after error: {e}")
                time.sleep(1)
                continue
            raise
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                print(f"Request retry {attempt + 1}/3: {e}")
                time.sleep(1)
                continue
            raise RuntimeError(f"Request to Groq API failed after retries: {e}")

    return {"error": "Unexpected fallback - no response obtained"}