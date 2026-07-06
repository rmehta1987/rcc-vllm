"""Health poll for ai-session vLLM servers.

Adapted from decrypto/slurm/ping_servers.py. Pings every discovered server with
a tiny chat request and prints reply latency. Used by ai_session.py's readiness
wait (it greps for 'Reply time') and by `ai_session status`.
"""

from __future__ import annotations

import time

from server import get_available_servers


def ping_all(verbose: bool = True) -> int:
    models = get_available_servers()
    ready = 0
    for model in models:
        for url in model["urls"]:
            try:
                from openai import OpenAI

                client = OpenAI(api_key="dummy_key", base_url=url)
                start = time.time()
                completion = client.chat.completions.create(
                    model=model["model_key"],
                    messages=[{"role": "user", "content": "Say 'ready' in one word."}],
                    max_tokens=8,
                )
                ready += 1
                if verbose:
                    print(
                        f" -  {model['model_key']} ({url}): "
                        f"{completion.choices[0].message.content!r} "
                        f"| Reply time: {time.time() - start:.2f} sec."
                    )
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"[!] {model['model_key']} ({url}): not responsive yet. {e}")

    if verbose:
        print("\nAVAILABLE SERVERS")
        for model in models:
            print(f" - {model['model_key']} ({model['model_id']}):")
            for url in model["urls"]:
                print(f"        - {url}")
    return ready


if __name__ == "__main__":
    ping_all(verbose=True)
