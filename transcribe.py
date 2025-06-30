import os
import asyncio
import io
import numpy as np
import soundfile as sf
from wyoming.server import AsyncServer
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.stt import Transcript, TranscriptResult
from nemo.collections.asr.models import EncDecCTCModel

PORT = int(os.getenv("PORT", 10300))  # Puerto por defecto 10300
MODEL_NAME = os.getenv("MODEL_NAME", "stt_es_conformer_ctc_large")  # Modelo por defecto

print(f"🔊 Cargando modelo {MODEL_NAME}...")
model = EncDecCTCModel.from_pretrained(MODEL_NAME)
print("✅ Modelo listo.")

class STTServer:
    def __init__(self):
        self.buffer = io.BytesIO()

    async def handle(self, client_reader, client_writer):
        server = AsyncServer(client_reader, client_writer)
        async for message in server:
            if isinstance(message, AudioStart):
                self.buffer = io.BytesIO()

            elif isinstance(message, AudioChunk):
                self.buffer.write(message.audio)

            elif isinstance(message, AudioStop):
                pcm = np.frombuffer(self.buffer.getvalue(), dtype=np.int16)
                wav_io = io.BytesIO()
                sf.write(wav_io, pcm, 16000, subtype='PCM_16', format='WAV')
                wav_io.seek(0)

                print("🎙️ Transcribiendo...")
                result = model.transcribe([wav_io])[0]
                print(f"📝 Resultado: {result}")

                await server.write(TranscriptResult(transcripts=[Transcript(text=result)]))

async def main():
    server = STTServer()
    await asyncio.start_server(server.handle, "0.0.0.0", PORT)
    print(f"🚪 Servidor Wyoming STT escuchando en puerto {PORT}...")
    await asyncio.Future()

asyncio.run(main())
