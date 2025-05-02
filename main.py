import os
import asyncio
import base64
import io
import threading
import keyboard
import numpy as np
import cv2
import pyaudio
import PIL.Image
import mss
from dotenv import load_dotenv
import pygame
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume, ISimpleAudioVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import speech_recognition as sr
from google import genai
from google.genai import types


load_dotenv()



async def safe_to_thread(func, *args, **kwargs):
    try:
        return await asyncio.to_thread(func, *args, **kwargs)
    except Exception as e:
        print(f"Error in {func.__name__}: {e}")


class AI:
    def __init__(self):
        self.ai_active = False
        self.toggle_event = threading.Event()
        self.video_mode = False # kinda wasteful
        self.wake_word_enabled = True
        self.model = "models/gemini-2.0-flash-live-001"
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.volume_reduction_percent = 80
        self.volume_reduction_blacklist = ["Discord.exe"]
        self.mic_reduction_percent = 90
        self.video_history_seconds = 5
        self.speaking_threshold = 4000

        self.prompt = """
These instructions are VERY important, follow them carefully.
Keep your responses short and concise unless necessary or asked to elaborate.
Like 2 or 3 sentences max.
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
Don't ask any questions, unless you absolutely need to ask a clarifying question.
Don't make small talk.
Commands given to you like this *this is a command* are commands you should follow but never mention them to the user. Treat them like they are invisible, not part of the conversation.
"""

        pygame.mixer.init()
        self.on_sound_obj = pygame.mixer.Sound("on.mp3")
        self.off_sound_obj = pygame.mixer.Sound("off.mp3")
        self.original_app_volumes = {}

        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.5
        self.recognizer.energy_threshold = self.speaking_threshold


        self.client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=self.api_key,
        )
        self.tools = [
            types.Tool(code_execution=types.ToolCodeExecution),
            types.Tool(google_search=types.GoogleSearch()),
        ]

        whats_up_command = "Say something short along the lines of 'What's up?' or 'What is it?"
        self.wake_words = [
            {"word": "need help", "command": whats_up_command},
            {"word": "need some help", "command": whats_up_command},
            {"word": "have a question", "command": whats_up_command},
            {"word": "quick question", "command": whats_up_command},
            {"word": "gemini", "command": None},
            {"word": "jemon", "command": None}
        ]

        self.gen_ai_config = types.LiveConnectConfig(
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
                )
            ),
            system_instruction=types.Content(
                parts=[
                    types.Part.from_text(
                        text=self.prompt,
                    )
                ],
                role="user",
            ),
            tools=self.tools
        )
        self.pya = pyaudio.PyAudio()
        self.audio_in_queue = None
        self.video_queue = None
        self.audio_queue = None
        self.session = None
        self.running = True
        self.audio_stream = None
        self.audio_out_stream = None
        self.ai_speaking = False
        self.user_speaking = False
        self.deactivate_counter = 0
        self.default_mic_info = None
        self.tasks = []



    def get_screen_input(self):
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


    async def watch_screen(self):
        while self.running:
            if not self.video_mode:
                await asyncio.sleep(1)
                continue
            try:
                frame = await safe_to_thread(self.get_screen_input)
                if frame:
                    if self.video_queue.full():
                        await self.video_queue.get()
                    await self.video_queue.put(frame)
                await asyncio.sleep(1.0)
            except Exception as e:
                print(f"Error in watch_screen: {e}")

    async def send_video(self):
        while self.running:
            try:
                if self.ai_active and not self.video_queue.empty():
                    msg = await self.video_queue.get()
                    await self.session.send(input=msg)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error in send_video: {e}")

    async def send_all_video(self):
        while not self.video_queue.empty():
            msg = await self.video_queue.get()
            await self.session.send(input=msg)

    async def send_audio(self):
        while self.running:
            try:
                if self.ai_active and not self.audio_queue.empty():
                    audio_msg = await self.audio_queue.get()
                    await self.session.send(input=audio_msg)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error in send_audio: {e}")

    async def listen_audio(self):
        kwargs = {"exception_on_overflow": False}
        while self.running:
            try:
                if self.ai_active and self.audio_stream and self.audio_stream.is_active():
                    data = await safe_to_thread(self.audio_stream.read, 1024, **kwargs)
                    await self.audio_queue.put({"data": data, "mime_type": "audio/pcm"})
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    volume = np.linalg.norm(audio_data)
                    self.user_speaking = volume > self.speaking_threshold
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                if "Stream closed" in str(e):
                    await self.activate_mic()
                else:
                    print(f"Error in listen_audio: {e}")

    async def adjust_mic_volume(self, reduce=True):
        devices = AudioUtilities.GetMicrophone()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None
        )
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        if reduce:
            volume.SetMasterVolumeLevelScalar(self.mic_reduction_percent / 100, None)
        else:
            volume.SetMasterVolumeLevelScalar(1.0, None)

    async def adjust_other_app_volumes(self, reduce=True):
        sessions = AudioUtilities.GetAllSessions()
        current_process_id = os.getpid()
        for session in sessions:
            if session.Process and session.Process.pid != current_process_id:
                volume = session._ctl.QueryInterface(ISimpleAudioVolume)
                app_name = session.Process.name() if session.Process else "Unknown"
                if app_name in self.volume_reduction_blacklist:
                    continue
                if reduce:
                    if app_name not in self.original_app_volumes:
                        self.original_app_volumes[app_name] = volume.GetMasterVolume()
                    reduced_volume = self.original_app_volumes[app_name] * (1 - self.volume_reduction_percent / 100)
                    volume.SetMasterVolume(reduced_volume, None)
                else:
                    if app_name in self.original_app_volumes:
                        volume.SetMasterVolume(self.original_app_volumes[app_name], None)

    async def activity_monitor(self):
        user_speaking_old = False
        send_image_cooldown = 3
        last_user_speaking_time = 0  # Track the last time the user was speaking
        while self.running:
            try:
                if self.ai_active:
                    current_time = asyncio.get_event_loop().time()
                    if not self.ai_speaking and not self.user_speaking:
                        self.deactivate_counter += 1
                        if self.deactivate_counter > 60:
                            await self.deactivate()
                    if self.ai_speaking or self.user_speaking:
                        self.deactivate_counter = 0

                    # Check if user changed from not speaking to speaking
                    if not user_speaking_old and self.user_speaking:
                        print("User is speaking...")
                        if current_time - last_user_speaking_time > send_image_cooldown:
                            frame = await safe_to_thread(self.get_screen_input)
                            if frame:
                                print("Sending image to AI...")
                                await self.session.send(input=frame)

                    # Update the last speaking time if the user is speaking
                    if self.user_speaking:
                        last_user_speaking_time = current_time

                    user_speaking_old = self.user_speaking
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error in activity_monitor: {e}")

    async def receive_audio(self):
        while self.running:
            try:
                if self.ai_active:
                    turn = self.session.receive()
                    async for response in turn:
                        if not self.ai_active:
                            break
                        if not self.ai_speaking and (response.data or response.text):
                            self.ai_speaking = True
                            await self.adjust_mic_volume(reduce=True)
                        if data := response.data:
                            self.audio_in_queue.put_nowait(data)
                            continue
                        if text := response.text:
                            print(text, end="")
                    while not self.audio_in_queue.empty():
                        self.audio_in_queue.get_nowait()
                    if self.ai_speaking:
                        self.ai_speaking = False
                        await self.adjust_mic_volume(reduce=False)
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error in receive_audio: {e}")

    async def activate_speaker(self):
        if self.audio_out_stream:
            await self.deactivate_speaker()
        self.audio_out_stream = await safe_to_thread(
            self.pya.open,
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
        )

    async def deactivate_speaker(self):
        if self.audio_out_stream:
            await safe_to_thread(self.audio_out_stream.stop_stream)
            await safe_to_thread(self.audio_out_stream.close)
            self.audio_out_stream = None

    async def play_audio(self):
        while self.running:
            try:
                if self.ai_active and self.audio_out_stream and not self.audio_in_queue.empty():
                    bytestream = await self.audio_in_queue.get()
                    if self.audio_out_stream and self.audio_out_stream.is_active():
                        await safe_to_thread(self.audio_out_stream.write, bytestream)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                if "Stream closed" in str(e) or "Unanticipated host error" in str(e) or "Stream not open" in str(e):
                    await self.activate_speaker()
                else:
                    print(f"Error in play_audio: {e}")


    async def empty_audio_queue(self):
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()
    
    async def empty_audio_in_queue(self):
        while not self.audio_in_queue.empty():
            self.audio_in_queue.get_nowait()


    async def activate_mic(self):
        if self.audio_stream:
            await self.deactivate_mic()
        self.default_mic_info = await safe_to_thread(self.pya.get_default_input_device_info)
        self.audio_stream = await safe_to_thread(
            self.pya.open,
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=self.default_mic_info["index"],
            frames_per_buffer=1024,
        )

    async def deactivate_mic(self):
        if self.audio_stream:
            await safe_to_thread(self.audio_stream.stop_stream)
            await safe_to_thread(self.audio_stream.close)
            self.audio_stream = None


    async def deactivate(self):
        self.ai_active = False
        await self.adjust_other_app_volumes(reduce=False)
        await self.adjust_mic_volume(reduce=False)
        await self.deactivate_mic()
        await self.deactivate_speaker()
        if self.off_sound_obj:
            await safe_to_thread(self.off_sound_obj.play)
        print("AI deactivated.")


    
    async def activate(self):
        self.deactivate_counter = 0
        self.user_speaking = False
        self.ai_speaking = False
        await self.empty_audio_queue()
        await self.empty_audio_in_queue()
        await self.send_all_video()
        await safe_to_thread(self.on_sound_obj.play)
        await self.adjust_other_app_volumes(reduce=True)
        await self.activate_mic()
        await self.activate_speaker()
        self.ai_active = True
        print("AI activated.")

    async def toggle_handler(self):
        while self.running:
            try:
                await safe_to_thread(self.toggle_event.wait)
                if self.ai_active:
                    await self.deactivate()
                else:
                    await self.activate()
                self.toggle_event.clear()
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error in toggle_handler: {e}")

    def setup_hotkey(self):
        def toggle_ai():
            self.toggle_event.set()
        keyboard.add_hotkey('alt+q', toggle_ai)

    def _recognize_wake_word(self):
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, phrase_time_limit=3, timeout=1)
            if not self.running or self.ai_active:
                return False
            text = self.recognizer.recognize_google(audio).lower()
            print(f"Recognized speech: {text}")
            for wake_word in self.wake_words:
                word = wake_word["word"].lower()
                if word in text:
                    print(f"Wake word detected: {word}")
                    return wake_word
        except sr.WaitTimeoutError:
            return False
        except Exception as e:
            if self.running:
                print(f"Error in _recognize_wake_word: {e}")
            return False

    async def listen_for_wake_word(self):
        while self.running:
            try:
                if not self.ai_active and self.wake_word_enabled:
                    wake_word = await safe_to_thread(self._recognize_wake_word)
                    if wake_word:
                        await self.activate()
                        if wake_word["command"]:
                            await asyncio.sleep(0.2)
                            await self.session.send(input=f"*{wake_word['command']}*", end_of_turn=True)
            except Exception as e:
                if self.running:
                    print(f"Error in listen_for_wake_word: {e}")
            await asyncio.sleep(0.1)

    async def cleanup(self):
        print(f"Cancelling {len(self.tasks)} tasks...")
        cancelled_count = 0
        for task in self.tasks:
            if not task.done():
                task.cancel()
                cancelled_count += 1
        print(f"{cancelled_count} tasks signalled for cancellation.")

        if self.tasks:
            try:
                print("Waiting for tasks to finish cancelling...")
                await asyncio.wait_for(asyncio.gather(*self.tasks, return_exceptions=True), timeout=2.0)
                print("Tasks cancellation processed.")
            except asyncio.TimeoutError:
                print("Timeout waiting for tasks to cancel. Some tasks may not have exited cleanly.")

        print("Unhooking keyboard...")
        try:
            await safe_to_thread(keyboard.unhook_all)
            print("Keyboard listener unhooked.")
        except Exception as e:
            print(f"Error unhooking keyboard: {e}")

        print("Terminating PyAudio...")
        if self.pya:
            try:
                await safe_to_thread(self.pya.terminate)
                print("PyAudio terminated.")
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")

        print("Quitting Pygame Mixer...")
        try:
            await safe_to_thread(pygame.mixer.quit)
            print("Pygame Mixer quit.")
        except Exception as e:
            print(f"Error quitting Pygame Mixer: {e}")

        print("Cleanup complete.")
        self.tasks = []

        
    async def run(self):
        self.tasks = []
        while self.running:
            await safe_to_thread(self.setup_hotkey)
            try:
                async with (
                    self.client.aio.live.connect(model=self.model, config=self.gen_ai_config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session = session
                    self.audio_in_queue = asyncio.Queue()
                    self.video_queue = asyncio.Queue(maxsize=self.video_history_seconds)
                    self.audio_queue = asyncio.Queue(maxsize=5)

                    self.tasks.append(tg.create_task(self.toggle_handler()))
                    self.tasks.append(tg.create_task(self.send_video()))
                    self.tasks.append(tg.create_task(self.send_audio()))
                    self.tasks.append(tg.create_task(self.listen_audio()))
                    self.tasks.append(tg.create_task(self.watch_screen()))
                    self.tasks.append(tg.create_task(self.receive_audio()))
                    self.tasks.append(tg.create_task(self.play_audio()))
                    self.tasks.append(tg.create_task(self.activity_monitor()))
                    self.tasks.append(tg.create_task(self.listen_for_wake_word()))
                    print("AI is running. Press Alt+Q to toggle.")
            except asyncio.CancelledError:
                print("Main run task cancelled (likely during shutdown).")
                self.running = False
            except Exception as e:
                print(f"Error in main run loop: {e}")
            finally:
                await self.cleanup()
                await asyncio.sleep(1.0)



if __name__ == "__main__":
    main = AI()
    asyncio.run(main.run())
    print("Program exited.")