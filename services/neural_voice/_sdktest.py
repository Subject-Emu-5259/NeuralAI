import asyncio, os, base64
from google import genai
from google.genai import types

async def main():
    key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=key, http_options={'api_version':'v1alpha'})
    cfg = types.LiveConnectConfig(response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Fenrir"))))
    try:
        async with client.aio.live.connect(model="gemini-2.5-flash-native-audio-latest", config=cfg) as session:
            print("CONNECT OK")
            # send a text turn
            await session.send(input=types.LiveClientContent(turns=[types.Content(parts=[types.Part(text="Hello, say hi back in one short sentence.")])]))
            got_audio=False; got_turn=False
            async for message in session:
                if message.server_content:
                    mt = message.server_content.model_turn
                    if mt:
                        for p in mt.parts:
                            if getattr(p,"inline_data",None):
                                got_audio=True
                    if message.server_content.turn_complete:
                        got_turn=True
                        break
            print("AUDIO", got_audio, "TURN", got_turn)
    except Exception as e:
        print("ERR:", repr(e))
asyncio.run(main())
