#!/usr/bin/env python3
"""Дёрнуть tool у stdio-MCP сервера напрямую: initialize → initialized → tools/call.

ЗАЧЕМ. Проверить сервер вне клиента: прогнать индексацию под конкретным
профилем, снять health_check, задать поисковый запрос — не перезапуская сессию
Claude и не полагаясь на то, какой бинарь она подняла при старте.

    RMC_EMBEDDING_PROFILE=local-gpu-bge \
      python3 tools/mcp_call.py ./rust-code-mcp index_codebase \
        '{"directory": "/path/to/repo", "force_reindex": true}' idx.err

Позиционные аргументы: бинарь, имя тула, JSON аргументов, и НЕОБЯЗАТЕЛЬНЫЙ файл
для stderr сервера (по умолчанию /dev/null). Ответ печатается как текст.

⚠ Ждёт РОВНО одну строку ответа, поэтому годится для одного вызова за запуск —
долгие туллы (индексация большого репо это десятки минут) держать в фоне.
"""
import json, subprocess, sys, time

binary, tool = sys.argv[1], sys.argv[2]
args = json.loads(sys.argv[3])

# 🚨 stderr — В ФАЙЛ, а не в PIPE, который никто не читает: сервер логирует
# в stderr, 64-килобайтный буфер пайпа переполняется на первой же крупной
# индексации, и сервер намертво блокируется на записи. Выглядит это не как
# ошибка, а как «индексация идёт долго»: процесс жив, потоки в futex_do_wait,
# I/O не движется (поймано на rust_app — 22 минуты тишины после BM25-фазы).
err = open(sys.argv[4] if len(sys.argv) > 4 else '/dev/null', 'w')
p = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=err, text=True, bufsize=1)

def send(obj):
    p.stdin.write(json.dumps(obj) + "\n")
    p.stdin.flush()

send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
      "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                 "clientInfo": {"name": "probe", "version": "0"}}})
print("init:", p.stdout.readline()[:120])
send({"jsonrpc": "2.0", "method": "notifications/initialized"})

t0 = time.time()
send({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
      "params": {"name": tool, "arguments": args}})
line = p.stdout.readline()
print(f"elapsed: {time.time()-t0:.1f}s")
try:
    resp = json.loads(line)
    for c in resp.get("result", {}).get("content", []):
        print(c.get("text", "")[:2000])
    if "error" in resp:
        print("ERROR:", resp["error"])
except Exception:
    print("raw:", line[:2000])
p.terminate()
