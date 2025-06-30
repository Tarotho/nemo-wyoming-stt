import asyncio
import io
import numpy as np
import soundfile as sf
from wyoming.server import AsyncServer
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.stt import Transcript, TranscriptResult
from nemo.collections.asr.models import EncDecCTCModel

print("🔊 Cargando modelo Conformer...")
model = EncDecCTCModel.from_pretrained("stt_es_conformer_ctc_large")
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
                # Convertir a WAV temporal para transcripción
                pcm = np.frombuffer(self.buffer.getvalue(), dtype=np.int16)
                wav_io = io.BytesIO()
                sf.write(wav_io, pcm, 16000, subtype='PCM_16', format='WAV')
                wav_io.seek(0)

                # Transcribir
                print("🎙️ Transcribiendo...")
                result = model.transcribe([wav_io])[0]
                print(f"📝 Resultado: {result}")

                # Enviar respuesta
                await server.write(TranscriptResult(transcripts=[Transcript(text=result)]))

async def main():
    server = STTServer()
    await asyncio.start_server(server.handle, "0.0.0.0", 10300)
    print("🚪 Servidor Wyoming STT escuchando en puerto 10300...")
    await asyncio.Future()  # run forever

asyncio.run(main())
