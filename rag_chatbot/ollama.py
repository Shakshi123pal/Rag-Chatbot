import asyncio
import threading
import socket
import requests


# ============================================================
#  Ollama Server Launcher (Your Original Code)
# ============================================================

def run_ollama_server():
    async def run_process(cmd):
        print('>>> starting', *cmd)
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def pipe(lines):
            async for line in lines:
                print(line.decode().strip())

            await asyncio.gather(
                pipe(process.stdout),
                pipe(process.stderr),
            )

        await asyncio.gather(pipe(process.stdout), pipe(process.stderr))

    async def start_ollama_serve():
        await run_process(['ollama', 'serve'])

    def run_async_in_thread(loop, coro):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(coro)
        loop.close()

    new_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=run_async_in_thread, args=(new_loop, start_ollama_serve()))
    thread.start()


def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect(('localhost', port))
            return True
        except ConnectionRefusedError:
            return False


# ============================================================
#  Ollama Chat Request Handler (NEW – FIX)
# ============================================================

def ollama_chat(prompt: str, model: str = "llama3.2:3b-instruct-q4_0"):
    """
    Send a prompt to the local Ollama model safely.
    Disables streaming & increases timeout to prevent 'Expected StreamingAgentChatResponse' error.
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        "stream": False  # important: disable streaming
    }

    try:
        print(f"[DEBUG] Sending request to Ollama ({len(prompt)} chars)")
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        # Some versions of Ollama return content inside 'message', others in 'messages'
        content = data.get("message", {}).get("content") or ""
        if not content and "messages" in data:
            content = data["messages"][-1].get("content", "")

        print("[INFO] ✅ Ollama response received successfully")
        return content.strip() if content else "⚠️ Empty response from Ollama."

    except requests.exceptions.Timeout:
        print("[ERROR] ❌ Ollama query timed out after 120s")
        return "⚠️ Ollama took too long to respond. Try smaller context."

    except Exception as e:
        print(f"[ERROR] Ollama query failed: {e}")
        return f"⚠️ Ollama query failed: {str(e)}"


# ============================================================
#  Warm-up Model Once at Startup
# ============================================================

def warmup_model():
    """Send a dummy query to warm up the Ollama model."""
    print("[INFO] 🔥 Warming up Ollama model...")
    try:
        result = ollama_chat("Hello!")
        print("[INFO] ✅ Ollama ready for queries.")
        return result
    except Exception as e:
        print(f"[WARN] Ollama warmup failed: {e}")
        return None
