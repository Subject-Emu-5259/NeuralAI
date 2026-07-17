import asyncio, json, websockets, base64
async def main():
    uri = "ws://127.0.0.1:5001/ws"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"type":"config","voice":"Andrew"}))
        await asyncio.sleep(1)
        await ws.send(json.dumps({"type":"text","data":"Hello, say the word test in one short sentence."}))
        got_audio=False; got_text=False; got_turn=False
        try:
            for _ in range(40):
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=12))
                if msg.get("type")=="audio": got_audio=True
                if msg.get("type")=="text": got_text=True
                if msg.get("type")=="turn_complete": got_turn=True; break
                if msg.get("type")=="error": print("WS ERROR:", msg); break
        except asyncio.TimeoutError:
            print("TIMEOUT")
        print("RESULT audio=%s text=%s turn_complete=%s" % (got_audio, got_text, got_turn))
asyncio.run(main())
