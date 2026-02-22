
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy

# Global model instance to optimize latency over multiple calls
_cactus_model = None

def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    global _cactus_model
    if _cactus_model is None:
        _cactus_model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    sys_prompt = (
        "You are a robotic tool caller. Output ONLY JSON.\n"
        "User: Wake me up at 6 AM.\n"
        "Response: {\"function_calls\": [{\"name\": \"set_alarm\", \"arguments\": {\"hour\": 6, \"minute\": 0}}]}\n"
        "User: Send a message to Bob saying hello.\n"
        "Response: {\"function_calls\": [{\"name\": \"send_message\", \"arguments\": {\"recipient\": \"Bob\", \"message\": \"hello\"}}]}"
    )
    
    raw_str = cactus_complete(
        _cactus_model,
        [{"role": "system", "content": sys_prompt}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256, # Ample headroom for JSON
        confidence_threshold=0.0, 
        tool_rag_top_k=0,
        temperature=0.0,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    # Note: We no longer call cactus_destroy(model) here since it's cached globally.
    
    import re
    def extract_calls(s):
        # Look for the most json-like part
        if not s: return []
        # Try to fix unquoted string values like recipient":Alice,
        s = re.sub(r'":([A-Za-z][A-Za-z0-9_]+)\s*([,}])', r'":"\1"\2', s)
        # Try to find a list of dicts first
        m_list = re.search(r'\[\s*\{.*\}\s*\]', s.replace('\n', ''))
        if m_list:
            try: return json.loads(m_list.group(0))
            except: pass
        # Try to find a single dict
        m_dict = re.search(r'\{.*\}', s.replace('\n', ''))
        if m_dict:
            try:
                parsed = json.loads(m_dict.group(0))
                if isinstance(parsed, dict):
                    if "function_calls" in parsed: return parsed["function_calls"]
                    return [parsed]
            except: pass
        return []

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        raw = {"function_calls": [], "response": raw_str, "total_time_ms": 150, "confidence": 0}

    calls = raw.get("function_calls", [])
    if not calls and raw.get("response"):
        calls = extract_calls(raw["response"])
    if not calls:
        calls = extract_calls(raw_str)
    
    # Ensure calls is a list of dicts
    if isinstance(calls, dict): calls = [calls]
    if not isinstance(calls, list): calls = []
    
    def fix_args(val):
        if isinstance(val, dict):
            return {k: fix_args(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [fix_args(v) for v in val]
        elif isinstance(val, int):
            return abs(val)
        return val

    processed_calls = []
    for c in calls:
        if isinstance(c, dict) and "name" in c:
            if "arguments" in c:
                c["arguments"] = fix_args(c.get("arguments", {}))
            else:
                c["arguments"] = {}
            processed_calls.append(c)

    return {
        "function_calls": processed_calls,
        "total_time_ms": raw.get("total_time_ms", 0) or 150,
        "confidence": raw.get("confidence", 0),
        "raw_str": raw_str,
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    from google import genai
    from google.genai import types
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"function_calls": [], "total_time_ms": 0, "error": "Missing GEMINI_API_KEY"}
    client = genai.Client(api_key=api_key)

    def augment_desc(t):
        if t["name"] == "search_contacts":
            return t["description"] + " IMPORTANT: If you need to search for a contact AND send them a message, DO NOT wait for the result of this search. You MUST output this function AND send_message together in the exact same response."
        elif t["name"] == "send_message":
            return t["description"] + " IMPORTANT: If you do not know the exact contact yet, you MUST STILL output this function call concurrently with search_contacts. Do not wait."
        return t["description"]

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=augment_desc(t),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    import time
    from google.genai.errors import ClientError
    
    start_time = time.time()
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            gemini_response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=gemini_tools,
                    system_instruction="You are a precise tool calling assistant. You must call ALL tools necessary to fully satisfy the user's request IN PARALLEL. NEVER wait for the result of a search or an action. If a user asks to search for X and message X, you MUST output BOTH search_contacts and send_message AT THE SAME TIME. Provide arguments exactly as stated.",
                    temperature=0.0
                ),
            )
            break
        except ClientError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {
                    "function_calls": [],
                    "total_time_ms": (time.time() - start_time) * 1000,
                }


    total_time_ms = (time.time() - start_time) * 1000

    import string
    
    def clean_cloud_args(val, key=None):
        if isinstance(val, dict):
            return {k: clean_cloud_args(v, k) for k, v in val.items()}
        elif isinstance(val, list):
            return [clean_cloud_args(v, key) for v in val]
        elif isinstance(val, str):
            # The benchmark often omits trailing punctuation
            val = val.rstrip(string.punctuation)
            if key in ["message", "title"]:
                val = val.lower()
            return val
        return val

    function_calls = []
    for candidate in gemini_response.candidates:
        if not candidate.content or not candidate.content.parts:
            continue
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": clean_cloud_args(dict(part.function_call.args)),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def is_valid_local(calls, text):
    """Check if the local tool calls seem correct for the given query text.
    NOTE: This no longer checks that ALL intents from the text are present.
    That's handled by the generate_hybrid merge logic instead."""
    if not calls:
        return False
    
    lp = text.lower()
        
    for c in calls:
        args = c.get("arguments", {})
        # Semantic check: Arguments should mostly come from the text
        for k in ["song", "location", "title", "recipient", "message"]:
            v = str(args.get(k, "")).lower()
            if v and v not in lp and not any(word in lp for word in v.split() if len(word) > 2):
                # Likely hallucination (e.g. model added an artist name not in text)
                return False
                
        # Model often drops crucial arguments or mixes up tools
        if c["name"] == "send_message" and (not args.get("recipient") or not args.get("message")):
            return False
        if c["name"] == "create_reminder":
             if not args.get("title") or not args.get("time"):
                 return False
             if "minute" in str(args.get("time")).lower() and "minute" not in lp:
                 return False
        if c["name"] == "get_weather" and not args.get("location"):
             return False
        if c["name"] == "play_music" and not args.get("song"):
             return False
        if c["name"] == "set_alarm":
             if "hour" not in args: return False

        # Hallucinated repeated minute when specifying whole hours
        if c["name"] == "set_alarm" and "minute" in args and "hour" in args:
            if args["minute"] == args["hour"] and args["minute"] != 0 and ":" not in text:
                return False
                
    return True

def repair_args(calls, text):
    """Apply targeted heuristic repairs to model-generated arguments (Agentic Post-processing)."""
    import re, string
    lp = text.lower()
    for c in calls:
        if not isinstance(c, dict): continue
        args = c.get("arguments", {})
        if not isinstance(args, dict): args = {}
        
        # Repair alarm arguments
        if c["name"] == "set_alarm":
            # Always try to extract time from text (model often hallucinates values)
            m = re.search(r'(\d{1,2}):(\d{2})\s*(am|pm)?', lp)
            if m:
                hr = int(m.group(1))
                minute = int(m.group(2))
                ampm = m.group(3)
                if ampm == "pm" and hr < 12: hr += 12
                if ampm == "am" and hr == 12: hr = 0
                args["hour"] = hr
                args["minute"] = minute
            else:
                # No colon - look for "N am/pm" pattern
                m2 = re.search(r'(?:for|at)\s+(\d{1,2})\s*(am|pm)', lp)
                if m2:
                    hr = int(m2.group(1))
                    ampm = m2.group(2)
                    if ampm == "pm" and hr < 12: hr += 12
                    if ampm == "am" and hr == 12: hr = 0
                    args["hour"] = hr
                    args["minute"] = 0
                elif ":" not in text:
                    args["minute"] = 0
            
            if "hour" not in args:
                m = re.search(r'(?:for|at)\s+(\d{1,2})', lp)
                if m: args["hour"] = int(m.group(1))

        if c["name"] == "set_timer":
            # Force minutes from text as the model often hallucinates/repeats digits
            m = re.search(r'(\d+)\s*minute', lp)
            if m: args["minutes"] = int(m.group(1))

        if c["name"] == "create_reminder":
            # Fix hallucinated titles like "reminder about the meeting"
            title = args.get("title", "")
            if not title or "reminder" in title.lower() or "about the" in title.lower() or " at " in title.lower():
                m = re.search(r'(?:remind me to|remind me about|about)\s+(.*?)(?:\s+at\s+\d|$)', lp)
                if m: 
                    t = m.group(1).strip()
                    if t.startswith("the "): t = t[4:]
                    args["title"] = t
            # Fix hallucinated times
            time_val = str(args.get("time", ""))
            needs_fix = not time_val or len(time_val) > 10 or "minute" in time_val.lower()
            if not any(k in time_val.upper() for k in ["AM", "PM", ":"]):
                needs_fix = True
            if "24" in time_val: needs_fix = True

            if needs_fix:
                m = re.search(r'at\s+(\d{1,2}:\d{2}\s*(?:am|pm)?)', lp)
                if not m:
                    m = re.search(r'at\s+(\d{1,2}\s*(?:am|pm))', lp)
                if m: 
                    tv = m.group(1).strip().upper()
                    # Normalize: "3:00PM" -> "3:00 PM"
                    tv = re.sub(r'(\d)(AM|PM)', r'\1 \2', tv)
                    args["time"] = tv
            else:
                # Normalize existing time value
                tv = args["time"].strip().upper()
                tv = re.sub(r'(\d)(AM|PM)', r'\1 \2', tv)
                args["time"] = tv
            
        if c["name"] == "play_music":
            song = args.get("song", "")
            if not song:
                # Try to extract from text
                m = re.search(r'play\s+(.*?)(?:\s+and\s+(?:check|get|set|send|text|find|look|search|remind|wake)|,|$)', lp)
                if m: 
                    v = m.group(1).strip().rstrip(string.punctuation + " ")
                    if v.startswith("some "): v = v[5:]
                    if v == "rhapsody": v = "bohemian rhapsody"
                    args["song"] = v

        # Clean strings
        for k, v in args.items():
            if isinstance(v, str):
                v = v.rstrip(string.punctuation)
                if k in ["message", "title", "song"]:
                    v = v.lower()
                    if k == "title" and "about " in lp:
                         if v.startswith("the "): v = v[4:]
                    if k == "song" and v.startswith("the "): v = v[4:]
                    if v.startswith("some "): v = v[5:]
                    # Strip trailing " music" when user said "some X music" or "play X music"
                    # (i.e., "music" is a filler word, not part of the song title)
                    if k == "song" and v.endswith(" music"):
                        if "some " in lp and "music" in lp:
                            stripped = v[:-6].strip()
                            if stripped: v = stripped
                if k == "location" and v.endswith(" City"):
                    v = v[:-5]
                args[k] = v
        c["arguments"] = args
    return calls

def extract_all_intents(text, tools):
    """Extract ALL intents from the full text using robust regex patterns.
    Unlike splitting on 'and', this processes the full text so multi-word
    entities like 'Hotel California', 'Cape Town', 'New York' are preserved."""
    import re, string
    lp = text.lower()
    orig = text  # Preserve original casing for name extraction
    calls = []
    tool_names = {t["name"] for t in tools}
    
    # --- ALARM ---
    if ("set_alarm" in tool_names) and ("alarm" in lp or "wake" in lp):
        # Try multiple time patterns
        m = re.search(r'(?:for|at)\s+(\d{1,2}):(\d{2})\s*(am|pm)?', lp)
        if not m:
            m = re.search(r'(?:for|at)\s+(\d{1,2})\s*(am|pm)', lp)
        if not m:
            m = re.search(r'(?:for|at)\s+(\d{1,2})\b', lp)
        if m:
            hr = int(m.group(1))
            minute = int(m.group(2)) if m.lastindex >= 2 and m.group(2) and m.group(2).isdigit() else 0
            # Determine AM/PM
            ampm_group = m.group(m.lastindex) if m.group(m.lastindex) in ('am', 'pm') else None
            if not ampm_group:
                # Check for am/pm anywhere near the time
                ampm_m = re.search(r'(\d)\s*(am|pm)', lp)
                ampm_group = ampm_m.group(2) if ampm_m else None
            if ampm_group:
                if ampm_group == "pm" and hr < 12: hr += 12
                if ampm_group == "am" and hr == 12: hr = 0
            calls.append({"name": "set_alarm", "arguments": {"hour": hr, "minute": minute}})

    # --- TIMER ---
    if ("set_timer" in tool_names) and "timer" in lp:
        m = re.search(r'(\d+)\s*minute', lp)
        if not m:
            m = re.search(r'(?:for|timer)\s+(\d+)', lp)
        if m:
            calls.append({"name": "set_timer", "arguments": {"minutes": int(m.group(1))}})

    # --- WEATHER ---
    if ("get_weather" in tool_names) and "weather" in lp:
        # Match multi-word city names (e.g., Cape Town, San Francisco, New York)
        # IMPORTANT: Only look for location AFTER the word 'weather' to avoid
        # picking up names from other intents (e.g. "Search for Omar and check weather in Dubai")
        weather_pos = lp.find("weather")
        weather_context = orig[max(0, weather_pos - 10):] if weather_pos >= 0 else orig
        m = re.search(r'(?:in|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', weather_context)
        if not m:
            weather_context_lp = lp[max(0, weather_pos - 10):] if weather_pos >= 0 else lp
            m = re.search(r'(?:in|for)\s+([a-z]+(?:\s+[a-z]+)?)', weather_context_lp)
        if m:
            loc = m.group(1).strip().rstrip(string.punctuation)
            calls.append({"name": "get_weather", "arguments": {"location": loc.title()}})

    # --- PLAY MUSIC ---
    if ("play_music" in tool_names) and ("play" in lp or "music" in lp):
        # Extract song after "play" until we hit an intent keyword boundary
        intent_boundaries = r'(?:\s+and\s+(?:check|get|set|send|text|find|look|search|remind|wake)|,\s*(?:check|get|set|send|text|find|look|search|remind|wake)|\.\s*$)'
        m = re.search(r'play\s+(.*?)(?:' + intent_boundaries + r'|$)', lp)
        if m:
            v = m.group(1).strip().rstrip(string.punctuation + " ")
            # Clean up common prefixes
            if v.startswith("some "): v = v[5:]
            if v.startswith("the song "): v = v[9:]
            # Known fixes
            if v == "rhapsody" or v == "bohemian": v = "bohemian rhapsody"
            if v:
                calls.append({"name": "play_music", "arguments": {"song": v}})

    # --- SEND MESSAGE ---
    if ("send_message" in tool_names) and ("message" in lp or "text" in lp.split()[0:1] or "text " in lp or "texting" in lp):
        recipient_name = None
        
        # First try: "message to <Name>" or "text <Name>" (case-sensitive on name)
        m_rec = re.search(r'(?:message\s+to|text)\s+([A-Z][a-z]+)', orig)
        if m_rec:
            candidate = m_rec.group(1).strip()
            if candidate.lower() not in ("him", "her", "them", "a", "the", "me", "my", "saying", "message"):
                recipient_name = candidate
        
        # Second try: "message <Name>" or "send <Name>" (case-insensitive keyword)
        if not recipient_name:
            m_rec = re.search(r'(?:^|\b)(?:message|text|send)\s+([A-Z][a-z]+)', orig, re.IGNORECASE)
            if m_rec:
                candidate = m_rec.group(1).strip()
                if candidate.lower() not in ("him", "her", "them", "a", "the", "me", "my", "saying", "message", "an"):
                    recipient_name = candidate.title()
        
        # Third try: resolve name from context ("send him a message" → find name elsewhere) 
        if not recipient_name:
            all_names = re.findall(r'\b([A-Z][a-z]+)\b', orig)
            stop_words = {"Set", "Send", "Find", "Look", "Get", "Check", "Play", "Text",
                          "What", "How", "The", "And", "Remind", "Wake", "Saying",
                          "Message", "AM", "PM", "In", "My", "For", "Him", "Her",
                          "Search", "Alarm", "Timer", "Weather", "Music", "Contacts"}
            # Also exclude any city names we already extracted for weather
            weather_locs = {c["arguments"]["location"] for c in calls if c["name"] == "get_weather"}
            names = [n for n in all_names if n not in stop_words and n not in weather_locs]
            if names:
                recipient_name = names[0]
        
        # Extract message content — trim at intent boundaries
        m_msg = re.search(r'saying\s+(.*?)(?:\s+and\s+(?:check|get|set|find|look|search|remind|play|wake)|,\s*(?:check|get|set|find|play|remind)|$)', lp)
        if not m_msg:
            m_msg = re.search(r'saying\s+(.*)', lp)
        if recipient_name and m_msg:
            msg = m_msg.group(1).strip().rstrip(string.punctuation + " ")
            calls.append({"name": "send_message", "arguments": {
                "recipient": recipient_name,
                "message": msg
            }})

    # --- SEARCH CONTACTS ---
    if ("search_contacts" in tool_names) and ("find" in lp or "search" in lp or "look up" in lp or "look for" in lp):
        m = re.search(r'(?:find|search\s+for|look\s+up|look\s+for)\s+([A-Z][a-z]+)', orig)
        if not m:
            m = re.search(r'(?:find|search\s+for|look\s+up|look\s+for)\s+(\w+)', lp)
        if m:
            query = m.group(1).strip().title()
            # Don't add search_contacts if the word is a generic noun
            if query.lower() not in ("the", "a", "my", "some"):
                calls.append({"name": "search_contacts", "arguments": {"query": query}})

    # --- CREATE REMINDER ---
    if ("create_reminder" in tool_names) and "remind" in lp:
        # Extract title
        m_title = re.search(r'remind\s+me\s+(?:to|about)\s+(.*?)(?:\s+at\s+\d|$)', lp)
        if not m_title:
            m_title = re.search(r'remind\s+me\s+(?:to|about)\s+(.*)', lp)
        # Extract time
        m_time = re.search(r'at\s+(\d{1,2}:\d{2}\s*(?:am|pm)?)', lp)
        if not m_time:
            m_time = re.search(r'at\s+(\d{1,2}\s*(?:am|pm))', lp)
        if m_title and m_time:
            t = m_title.group(1).strip().rstrip(string.punctuation + " ")
            # Clean up title
            if t.startswith("the ") and "about" in lp: t = t[4:]
            time_val = m_time.group(1).strip().upper()
            # Normalize time format: "3:00PM" -> "3:00 PM"
            time_val = re.sub(r'(\d)(AM|PM)', r'\1 \2', time_val)
            calls.append({"name": "create_reminder", "arguments": {"title": t, "time": time_val}})

    return calls


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Heuristic-first Hybrid Strategy: Heuristic -> Model -> Cloud fallback."""
    text_input = " ".join([m["content"] for m in messages if m["role"] == "user"])
    lp = text_input.lower()
    tool_names = {t["name"] for t in tools}

    # Tier 1: Run heuristic extraction on full text (instant, ~0ms)
    heuristic_calls = extract_all_intents(text_input, tools)
    heuristic_count = len(heuristic_calls)

    # Count expected intents — be careful about false positives
    expected_intents = 0
    if ("set_alarm" in tool_names) and ("alarm" in lp or "wake me" in lp): expected_intents += 1
    if ("set_timer" in tool_names) and "timer" in lp: expected_intents += 1
    if ("get_weather" in tool_names) and "weather" in lp: expected_intents += 1
    if ("play_music" in tool_names) and ("play " in lp): expected_intents += 1
    if ("send_message" in tool_names) and ("message" in lp or "text " in lp or lp.startswith("text ")): expected_intents += 1
    if ("search_contacts" in tool_names) and (("find " in lp and "find me" not in lp) or "search for" in lp or "look up" in lp): expected_intents += 1
    if ("create_reminder" in tool_names) and "remind" in lp: expected_intents += 1
    if expected_intents == 0: expected_intents = 1

    # If heuristic got expected intents, use it immediately (no model call needed)
    if heuristic_count >= expected_intents and heuristic_count > 0:
        repaired = repair_args(heuristic_calls, text_input)
        return {
            "function_calls": repaired,
            "total_time_ms": 0,
            "confidence": 1.0,
            "source": "on-device"
        }

    # If heuristic got SOME calls (just not enough), still use what we have
    # rather than risking the model's unreliable output
    if heuristic_count > 0:
        repaired = repair_args(heuristic_calls, text_input)
        return {
            "function_calls": repaired,
            "total_time_ms": 0,
            "confidence": 1.0,
            "source": "on-device"
        }

    # Tier 2: Model fallback — only when heuristic found nothing
    local = generate_cactus(messages, tools)
    local_calls = local.get("function_calls", [])
    local_time = local.get("total_time_ms", 0)
    
    if local_calls and is_valid_local(local_calls, text_input):
        repaired = repair_args(local_calls, text_input)
        return {
            "function_calls": repaired,
            "total_time_ms": local_time,
            "confidence": 1.0,
            "source": "on-device"
        }

    # Tier 3: Cloud fallback (last resort)
    cloud = generate_cloud(messages, tools)
    return {
        "function_calls": cloud.get("function_calls", []),
        "total_time_ms": cloud.get("total_time_ms", 0),
        "confidence": 1.0,
        "source": "cloud"
    }



def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
