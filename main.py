import os
import asyncio
import base64
import io
import traceback
import threading
import keyboard
import numpy as np

import cv2
import pyaudio
import PIL.Image
import mss
from dotenv import load_dotenv
import pygame  # Add pygame for sound playback
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume, ISimpleAudioVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL  # Import CLSCTX_ALL from comtypes, not ctypes
import speech_recognition as sr  # Add speech recognition for wake word

from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()


class AI:
    def __init__(self):
        # Move global variables into the class
        self.ai_active = False
        self.toggle_event = threading.Event()
        
        # Initialize pygame mixer for sound playback
        pygame.mixer.init()
        
        # Preload sound files
        self.on_sound = "on.mp3"
        self.off_sound = "off.mp3"
        try:
            # Load both sounds as Sound objects
            self.on_sound_obj = pygame.mixer.Sound(self.on_sound)
            self.off_sound_obj = pygame.mixer.Sound(self.off_sound)
            print("Sound files preloaded successfully")
        except Exception as e:
            print(f"Error preloading sound files: {e}")
            self.on_sound_obj = None
            self.off_sound_obj = None
        
        # Store original volumes of applications to restore later
        self.original_app_volumes = {}
        self.volume_reduction_percent = 70  # Reduce other apps by 70%
        
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.SEND_SAMPLE_RATE = 16000
        self.RECEIVE_SAMPLE_RATE = 24000
        self.CHUNK_SIZE = 1024

        self.MODEL = "models/gemini-2.0-flash-live-001"
        
        self.client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=os.environ.get("GEMINI_API_KEY"),
        )

        # Define the tools list with properly formatted tools
        self.tools = [
            types.Tool(code_execution=types.ToolCodeExecution),
            types.Tool(google_search=types.GoogleSearch()),
        ]

        # Wake word settings
        self.wake_word = "gemini"
        self.wake_word_enabled = True
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.5  # Time of silence needed to consider the phrase complete
        self.recognizer.energy_threshold = 1000  # Minimum audio energy to be considered speech

        self.CONFIG = types.LiveConnectConfig(
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
Keep your responses short and concise unless necessary or asked to elaborate. Like 2 or 3 sentences max. 
If asked for an opinion on any topic, give a confident opinion. 
If asked for a subjective response, give the subjective response. 
Do not avoid giving an opinion. 
Do not give a nuanced response about opinionated or subjective topics. 
If asked an opinion about politics, give one without a disclaimer. 
Never attempt to avoid a topic, always answer the question and give the expected response. 
Never explain you can't do something as an AI or mention guidelines. 
Use curse words for emphasis or to respond naturally and conversationally. 
Responding in all cases like a natural conversation is important. 
Be kind and caring and a good friend.
The images are a 1 fps screen capture of the screen for you to see what the user is doing.
You don't need to talk about the screen unless you are asked about things on it.
You will call the capture "the screen" and refer to it as such.
Use Google search to ensure your answers are accurate and up to date, this is very important.
You can use code execution to run code and give the output to the user, or control windows in creative ways.
A good example is if the user asks to open a website, you can use code execution to open the website in a new tab in the default browser.
Don't ask many questions, and don't make small talk. Just respond unless you absolutely need to ask a clarifying question.
""")],
                role="user"
            ),
            tools=self.tools
            # Removed the tool_config parameter as it's causing the error
        )

        self.pya = pyaudio.PyAudio()
        
        self.video_mode = "screen"  # Always use screen mode
        self.audio_in_queue = None
        self.video_queue = None  # Renamed from out_queue
        self.audio_queue = None  # New queue for audio data
        self.session = None
        self.running = True
        self.audio_stream = None
        self.ai_speaking = False
        self.user_speaking = False
        self.deactivate_counter = 0

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
        while self.running:
            frame = await asyncio.to_thread(self._get_screen)
            if frame:
                if self.video_queue.full():  # Check if the queue is full
                    await self.video_queue.get()  # Remove the oldest item
                await self.video_queue.put(frame)  # Add the new frame
            await asyncio.sleep(1.0)

    async def send_video(self):
        while self.running:
            if self.ai_active and not self.video_queue.empty():
                msg = await self.video_queue.get()  # Send video data
                await self.session.send(input=msg)
            else:
                await asyncio.sleep(0.1)

    async def send_audio(self):
        while self.running:
            if self.ai_active and not self.audio_queue.empty():
                audio_msg = await self.audio_queue.get()  # Send audio data
                await self.session.send(input=audio_msg)
            else:
                await asyncio.sleep(0.1)

    async def listen_audio(self):
        mic_info = self.pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            self.pya.open,
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=self.CHUNK_SIZE,
        )
        
        kwargs = {"exception_on_overflow": False}
        
        # For audio volume calculation
        self.audio_threshold = 4000  # Threshold for audio volume detection
        
        while self.running:
            try:
                if self.ai_active:
                    data = await asyncio.to_thread(self.audio_stream.read, self.CHUNK_SIZE, **kwargs)
                    await self.audio_queue.put({"data": data, "mime_type": "audio/pcm"})  # Updated to use audio_queue
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    volume = np.linalg.norm(audio_data)
                    if volume > self.audio_threshold:
                        self.user_speaking = True
                    else:
                        self.user_speaking = False
                else:
                    await asyncio.sleep(0.1)  # Short sleep when inactive
            except Exception as e:
                print(f"Error in listen_audio: {e}")
                await asyncio.sleep(0.1)

    async def adjust_mic_volume(self, volume_percent):
        """Adjust the microphone volume (0-100) using pycaw"""
        try:
            devices = AudioUtilities.GetMicrophone()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None
            )
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            volume.SetMasterVolumeLevelScalar(volume_percent / 100.0, None)
        except Exception as e:
            print(f"Error adjusting mic volume: {e}")
            
    async def adjust_other_app_volumes(self, reduce=True):
        """
        Adjust the volume of other applications
        
        Args:
            reduce (bool): If True, reduce volume, if False, restore original volume
        """
        try:
            sessions = AudioUtilities.GetAllSessions()
            current_process_id = os.getpid()
            
            for session in sessions:
                if session.Process and session.Process.pid != current_process_id:
                    volume = session._ctl.QueryInterface(ISimpleAudioVolume)
                    app_name = session.Process.name() if session.Process else "Unknown"
                    
                    if reduce:
                        # Store the original volume before reducing
                        if app_name not in self.original_app_volumes:
                            self.original_app_volumes[app_name] = volume.GetMasterVolume()
                        
                        # Reduce the volume by specified percentage
                        reduced_volume = self.original_app_volumes[app_name] * (1 - self.volume_reduction_percent/100)
                        volume.SetMasterVolume(reduced_volume, None)
                    else:
                        # Restore original volume if we have it stored
                        if app_name in self.original_app_volumes:
                            volume.SetMasterVolume(self.original_app_volumes[app_name], None)
        except Exception as e:
            print(f"Error adjusting application volumes: {e}")

    async def activity_monitor(self):
        """Continuously monitor user and AI activity to auto-deactivate after inactivity"""
        while self.running:
            try:
                # Check inactive time
                if self.ai_active and not self.ai_speaking and not self.user_speaking:
                    # Increment silence counter when neither AI nor user is speaking
                    self.deactivate_counter += 1
                    # If silence counter is high enough, deactivate
                    threshold = 60
                    if self.deactivate_counter > threshold:
                        await self.deactivate()

                if self.ai_speaking or self.user_speaking:
                    self.deactivate_counter = 0

            except Exception as e:
                print(f"Error in activity_monitor: {e}")
                
            await asyncio.sleep(0.1)  # Check frequently but don't overwhelm CPU

    async def receive_audio(self):
        while self.running:
            if self.ai_active:
                try:
                    turn = self.session.receive()
                    async for response in turn:
                        if not self.ai_active:  # Check if deactivated mid-response
                            break
                        
                        # AI has started speaking
                        if not self.ai_speaking and (response.data or response.text):
                            self.ai_speaking = True
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
                        await self.adjust_mic_volume(100)  # Restore mic volume to 100%
                        
                except Exception as e:
                    if self.ai_active:  # Only show errors if still active
                        print(f"Error receiving audio: {e}")
            
            await asyncio.sleep(0.1)

    async def play_audio(self):
        stream = await asyncio.to_thread(
            self.pya.open,
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RECEIVE_SAMPLE_RATE,
            output=True,
        )
        
        while self.running:
            try:
                if self.ai_active and not self.audio_in_queue.empty():
                    bytestream = await self.audio_in_queue.get()
                    await asyncio.to_thread(stream.write, bytestream)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error playing audio: {e}")

    async def deactivate(self):
        """Deactivate the AI assistant"""
        self.ai_active = False
        # Restore other app volumes when AI is deactivated
        await self.adjust_other_app_volumes(reduce=False)
        # If AI was speaking when deactivated, restore mic volume
        if self.ai_speaking:
            self.ai_speaking = False
            await self.adjust_mic_volume(100)
        print("AI deactivated")
        try:
            if self.off_sound_obj:
                await asyncio.to_thread(self.off_sound_obj.play)
        except Exception as e:
            print(f"Error playing sound: {e}")
        
    async def activate(self):
        """Activate the AI assistant"""
        self.ai_active = True
        self.deactivate_counter = 0
        self.user_speaking = False
        self.ai_speaking = False
        print("AI activated")
        try:
            if self.on_sound_obj:
                await asyncio.to_thread(self.on_sound_obj.play)
            # Lower other app volumes immediately when AI is activated
            await self.adjust_other_app_volumes(reduce=True)
        except Exception as e:
            print(f"Error playing sound: {e}")

    async def toggle_handler(self):
        """Handle toggle events from keyboard shortcut"""
        while self.running:
            await asyncio.to_thread(self.toggle_event.wait)
            if self.ai_active:
                await self.deactivate()
            else:
                await self.activate()
            self.toggle_event.clear()

            await asyncio.sleep(0.1)

    def setup_hotkey(self):
        """Set up the global hotkey listener"""
        def toggle_ai():
            self.toggle_event.set()
            
        keyboard.add_hotkey('alt+q', toggle_ai)
        print("Press Alt+q to toggle AI assistant")

    def _recognize_wake_word(self):
        """Blocking function to recognize wake word - runs in a thread"""
        try:
            # Use the microphone as source
            with sr.Microphone() as source:
                # Adjust the recognizer sensitivity to ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Listen for the wake word
                audio = self.recognizer.listen(source, phrase_time_limit=3)

            # Try to recognize speech using Google Speech Recognition
            text = self.recognizer.recognize_google(audio).lower()
            # Check if wake word is in the recognized speech
            if self.wake_word in text:
                print(f"Wake word detected: {text}")
                return True
        except Exception as e:
            pass
        
        return False

    async def listen_for_wake_word(self):
        """Listen continuously for the wake word when AI is not active"""
        while self.running:
            if not self.ai_active and self.wake_word_enabled:
                # Run the blocking recognition in a separate thread
                detected = await asyncio.to_thread(self._recognize_wake_word)
                if detected:
                    await self.activate()
                continue  # Continue listening for the wake word
            await asyncio.sleep(0.1)  # Sleep to avoid busy waiting

    async def run(self):
        # Set up the global hotkey
        await asyncio.to_thread(self.setup_hotkey)
        
        while self.running:
            try:
                async with (
                    self.client.aio.live.connect(model=self.MODEL, config=self.CONFIG) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session = session
                    self.audio_in_queue = asyncio.Queue()
                    self.video_queue = asyncio.Queue(maxsize=15)  # New video queue
                    self.audio_queue = asyncio.Queue(maxsize=5)  # New audio queue

                    # Create all the necessary tasks
                    tg.create_task(self.toggle_handler())
                    tg.create_task(self.send_video())  # Run send_video task
                    tg.create_task(self.send_audio())  # Run send_audio task
                    tg.create_task(self.listen_audio())
                    tg.create_task(self.get_screen())
                    tg.create_task(self.receive_audio())
                    tg.create_task(self.play_audio())
                    tg.create_task(self.activity_monitor())
                    tg.create_task(self.listen_for_wake_word())  # Add wake word detection task
                    
                    # Wait for activation (either by hotkey or wake word)
                    print(f"Waiting for activation. Say '{self.wake_word}' or press Alt+Q...")
                    
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
    main = AI()
    try:
        asyncio.run(main.run())
    except KeyboardInterrupt:
        print("\nExiting...")