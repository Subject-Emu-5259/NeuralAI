import asyncio, os
from google import genai
from google.genai import types

async def main():
    key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=key, http_options={'api_version':'v1alpha'})
    cfg = types.LiveConnectConfig(response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Fenrir"))))
    async with client.aio.live.connect(model="gemini-2.5-flash-native-audio-latest", config=cfg) as session:
        print("CONNECT OK")
        await session.send_client_content(turns=[types.Content(parts=[types.Part(text="Say hi in exactly three words.")])])
        got_audio=False; got_turn=False
        while True:
            message = await session.receive()
            if message.server_content:
                mt = message.server_content.model_turn
                if mt:
                    for p in mt.parts:
                        if getattr(p,"inline_data",None): got_audio=True
                if message.server_content.turn_complete:
                    got_turn=True; break
            if getattr(message,"tool_call",None): pass
        print("AUDIO", got_audio, "TURN", got_turn)
asyncio.run(main())
