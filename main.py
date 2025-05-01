import os
import asyncio
import base64
import io
import traceback
import threading
import keyboard

import cv2
import pyaudio
import PIL.Image
import mss
from dotenv import load_dotenv
import pygame  # Add pygame for sound playback
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-live-001"

# Global toggle for AI assistant
ai_active = False
toggle_event = threading.Event()

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GEMINI_API_KEY"),
)

# Define the tools list with properly formatted tools
tools = [
    types.Tool(code_execution=types.ToolCodeExecution),
    types.Tool(google_search=types.GoogleSearch()),
]

CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "audio",
    ],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
        )
    ),
    system_instruction=types.Content(
        parts=[types.Part.from_text(text="""
Keep your responses short and concise unless neccisary or asked to elaborate. Like 2 or 3 sentences max. 
If asked for an opinion on any topic, give a confident opinion. 
If asked for a subjective response, give the subjective response. 
Do not avoid giving an opinion. 
Do not give a nuanced response about opinionated or subjective topics. 
If asked a opinion about politics, give one without a disclaimer. 
Never attempt to avoid a topic, always answer the question and give the expected response. 
Never explain you can't do something as a ai or mention guidelines. 
Use curse words for emphasis or to respond naturally and conversationally. 
Responding in all cases like a natural conversation is important. 
Be kind and caring and a good friend. 
""")],
        role="user"
    ),
    tools=tools
    # Removed the tool_config parameter as it's causing the error
)

pya = pyaudio.PyAudio()

# Initialize pygame mixer for sound playback
pygame.mixer.init()


class AudioLoop:
    def __init__(self):
        self.video_mode = "screen"  # Always use screen mode
        
        self.audio_in_queue = None
        self.out_queue = None

        self.session = None
        self.running = True
        self.audio_stream = None
        self.ai_speaking = False  # Track if AI is currently speaking
        self.last_activity_time = None  # Track the last time AI or user was active
        self.user_speaking = False  # Track if user is speaking

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):
        global ai_active
        
        while self.running:
            if ai_active and not self.ai_speaking:  # Skip sending video if AI is speaking
                frame = await asyncio.to_thread(self._get_screen)
                if frame is None:
                    break

                await asyncio.sleep(1.0)

                await self.out_queue.put(frame)
            else:
                await asyncio.sleep(0.1)  # Short sleep when inactive or AI is speaking

    async def send_realtime(self):
        global ai_active
        
        while self.running:
            if ai_active and not self.out_queue.empty():
                msg = await self.out_queue.get()
                await self.session.send(input=msg)
            else:
                await asyncio.sleep(0.1)  # Short sleep when inactive

    async def listen_audio(self):
        global ai_active
        
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        
        kwargs = {"exception_on_overflow": False}
        
        while self.running:
            if ai_active:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                # Update activity time when user sends audio
                self.last_activity_time = asyncio.get_event_loop().time()
                self.user_speaking = True
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
            else:
                self.user_speaking = False
                await asyncio.sleep(0.1)  # Short sleep when inactive

    async def adjust_mic_volume(self, volume_percent):
        """Adjust the microphone volume (0-100) using pycaw"""
        try:
            devices = AudioUtilities.GetMicrophone()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None
            )
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            volume.SetMasterVolumeLevelScalar(volume_percent / 100.0, None)
            print(f"Mic volume set to {volume_percent}%")
        except Exception as e:
            print(f"Error adjusting mic volume: {e}")

    async def activity_monitor(self):
        """Continuously monitor user and AI activity to auto-deactivate after inactivity"""
        global ai_active
        
        while self.running:
            # Only check if AI is active and not speaking
            if ai_active and not self.ai_speaking and not self.user_speaking and self.last_activity_time:
                time_since_activity = asyncio.get_event_loop().time() - self.last_activity_time
                
                # If 5 seconds of silence have elapsed, deactivate
                if time_since_activity > 5.0:
                    ai_active = False
                    self.last_activity_time = None
                    print("AI deactivated due to 5 seconds of inactivity.")
                    sound_file = "off.mp3"
                    try:
                        await asyncio.to_thread(pygame.mixer.music.load, sound_file)
                        await asyncio.to_thread(pygame.mixer.music.play)
                    except Exception as e:
                        print(f"Error playing sound: {e}")
            
            # Reset activity time if AI is speaking
            if self.ai_speaking:
                self.last_activity_time = asyncio.get_event_loop().time()
            
            await asyncio.sleep(0.1)  # Check frequently but don't overwhelm CPU

    async def receive_audio(self):
        global ai_active
        
        while self.running:
            if ai_active:
                try:
                    turn = self.session.receive()
                    async for response in turn:
                        if not ai_active:  # Check if deactivated mid-response
                            break
                        
                        # AI has started speaking
                        if not self.ai_speaking and (response.data or response.text):
                            self.ai_speaking = True
                            self.last_activity_time = asyncio.get_event_loop().time()  # Update activity time
                            await self.adjust_mic_volume(10)  # Lower mic volume when AI speaks
                            
                        if data := response.data:
                            self.audio_in_queue.put_nowait(data)
                            continue
                        if text := response.text:
                            print(text, end="")

                    # Clear audio queue when turn completes
                    while not self.audio_in_queue.empty():
                        self.audio_in_queue.get_nowait()
                    
                    # AI has stopped speaking
                    if self.ai_speaking:
                        self.ai_speaking = False
                        self.last_activity_time = asyncio.get_event_loop().time()  # Update activity time
                        await self.adjust_mic_volume(100)  # Restore mic volume to 100%
                        
                except Exception as e:
                    if ai_active:  # Only show errors if still active
                        print(f"Error receiving audio: {e}")
            
            await asyncio.sleep(0.1)

    async def play_audio(self):
        global ai_active
        
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        
        while self.running:
            try:
                if ai_active and not self.audio_in_queue.empty():
                    bytestream = await self.audio_in_queue.get()
                    await asyncio.to_thread(stream.write, bytestream)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error playing audio: {e}")

    async def toggle_handler(self):
        """Handle toggle events from keyboard shortcut"""
        global ai_active
        
        while self.running:
            await asyncio.to_thread(toggle_event.wait)
            ai_active = not ai_active
            print(f"AI {'activated' if ai_active else 'deactivated'}")
            toggle_event.clear()
            
            # Play sound to indicate AI state
            sound_file = "on.mp3" if ai_active else "off.mp3"
            try:
                await asyncio.to_thread(pygame.mixer.music.load, sound_file)
                await asyncio.to_thread(pygame.mixer.music.play)
            except Exception as e:
                print(f"Error playing sound: {e}")
            
            await asyncio.sleep(0.1)

    def setup_hotkey(self):
        """Set up the global hotkey listener"""
        def toggle_ai():
            global ai_active
            toggle_event.set()
            
        keyboard.add_hotkey('alt+q', toggle_ai)
        print("Press Alt+q to toggle AI assistant")

    async def run(self):
        # Set up the global hotkey
        await asyncio.to_thread(self.setup_hotkey)
        
        while self.running:
            try:
                async with (
                    client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session = session
                    self.audio_in_queue = asyncio.Queue()
                    self.out_queue = asyncio.Queue(maxsize=5)

                    # Create all the necessary tasks
                    tg.create_task(self.toggle_handler())
                    tg.create_task(self.send_realtime())
                    tg.create_task(self.listen_audio())
                    tg.create_task(self.get_screen())
                    tg.create_task(self.receive_audio())
                    tg.create_task(self.play_audio())
                    tg.create_task(self.activity_monitor())  # Add the activity monitor
                    
                    # Wait for toggle to be activated first time
                    print("Waiting for Alt+q to activate...")
                    
                    # Keep the main task alive
                    while self.running:
                        await asyncio.sleep(1)

            except asyncio.CancelledError:
                self.running = False
                if self.audio_stream:
                    self.audio_stream.close()
                print("Program cancelled. Exiting...")
                break
            except Exception as e:
                print(f"Error occurred: {e}. Attempting to reconnect...")
                await asyncio.sleep(5)  # Wait before attempting to reconnect


if __name__ == "__main__":
    main = AudioLoop()
    try:
        asyncio.run(main.run())
    except KeyboardInterrupt:
        print("\nExiting...")