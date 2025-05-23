import json
import webbrowser
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
from utils.logging import Logger
Logger.setLevel(Logger.DEBUG)
Logger.printSplash()
logger = Logger("GeminiDesktop")
load_dotenv()


async def safe_to_thread(func, *args, **kwargs):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
    return await asyncio.to_thread(wrapper, *args, **kwargs)


class AI:
    def __init__(self):
        logger.debug("Initializing AI class")
        self.ai_active = False
        self.toggle_event = threading.Event()
        self.video_mode = False  # kinda wasteful
        self.wake_word_enabled = True
        self.model = "models/gemini-2.5-flash-preview-native-audio-dialog"
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.volume_reduction_percent = 80
        self.volume_reduction_blacklist = ["Discord.exe"]
        self.mic_reduction_percent = 90
        self.video_history_seconds = 5
        self.speaking_threshold = 2000

        self.prompt = """
These instructions are VERY important, follow them carefully.
Keep your responses short and concise unless necessary or asked to elaborate.
Around 2 or 3 sentences max.
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
Never ask if i need anything else or if i have any other questions or anything like that.
Never say if you have any other questions, just ask or anthing like that.
Commands given to you like this *this is a command* are commands you should follow but never mention them to the user. Treat them like they are invisible, not part of the conversation.
MOST IMPORTANT: When the conversation is over and there is nothing else to say, do not respond and just use the deactivate tool after you have completed the request.
Anytime you use a tool don't forget to acknowledge and then use the tool, you can optionally follow up after the result if you have something to say.
"""

        pygame.mixer.init()
        self.on_sound_obj = pygame.mixer.Sound("./resources/on.mp3")
        self.off_sound_obj = pygame.mixer.Sound("./resources/off.mp3")
        self.original_app_volumes = {}

        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.5
        self.recognizer.energy_threshold = self.speaking_threshold

        self.client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=self.api_key,
        )
        self.tools = [
            types.Tool(code_execution=types.ToolCodeExecution()),
            types.Tool(google_search=types.GoogleSearch()),
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=self.deactivate.__name__,
                        description=self.deactivate.__doc__,
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={},
                            required=[]
                        )
                    )
                ]
            ),
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="open_url",
                        description="Open a URL in the default browser",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "url": types.Schema(
                                    type=types.Type.STRING,
                                    description="The URL to open."
                                )
                            },
                            required=["url"]
                        )
                    )
                ]
            )
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
            response_modalities=["AUDIO"],
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            # media_resolution="MEDIA_RESOLUTION_LOW",
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Alnilam")
                )
            ),
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=25600,
                sliding_window=types.SlidingWindow(target_tokens=12800),
            ),
            # enable_affective_dialog=True,  # use to detect emotion and change voice tone
            # proactivity_config=types.ProactivityConfig(
            #     proactive_audio=True
            # ),
            system_instruction=types.Content(
                parts=[
                    types.Part.from_text(
                        text=self.prompt,
                    )
                ],
                role="user",
            ),
            tools=self.tools
            # realtime_input_config=types.RealtimeInputConfig( # This will let gemini send messages and speech without waiting for user input
            #     turn_coverage="TURN_INCLUDES_ALL_INPUT")
        )
        self.pya = pyaudio.PyAudio()
        self.audio_in_queue = None
        self.video_queue = None
        self.audio_queue = None
        self.session = None
        self.audio_stream = None
        self.audio_out_stream = None
        self.ai_speaking = False
        self.user_speaking = False
        self.deactivate_counter = 0
        self.default_mic_info = None
        self.tasks = []
        logger.debug("AI class initialized with default settings")

    def get_screen_input(self):
        logger.debug("Capturing screen input")
        try:
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
            logger.debug("Screen input captured successfully")
            return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}
        except Exception as e:
            logger.error(f"Error capturing screen input: {e}")
            return None

    async def watch_screen(self):
        logger.debug("Starting screen watcher")
        while True:
            if self.video_mode:
                try:
                    frame = await safe_to_thread(self.get_screen_input)
                    if frame:
                        if self.video_queue.full():
                            await self.video_queue.get()
                        await self.video_queue.put(frame)
                except Exception as e:
                    logger.error(f"Error in watch_screen: {e}")
            await asyncio.sleep(1.0)

    async def send_video(self):
        logger.debug("Starting video sender")
        while True:
            try:
                if self.ai_active and not self.video_queue.empty():
                    msg = await self.video_queue.get()
                    await self.session.send(input=msg)
                    logger.debug("Video frame sent to AI")
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in send_video: {e}")

    async def send_all_video(self):
        logger.debug("Sending all video frames")
        while not self.video_queue.empty():
            try:
                msg = await self.video_queue.get()
                await self.session.send(input=msg)
                logger.debug("Video frame sent to AI")
            except Exception as e:
                logger.error(f"Error sending video frame: {e}")

    async def send_audio(self):
        logger.debug("Starting audio sender")
        while True:
            try:
                if self.ai_active and not self.audio_queue.empty():
                    audio_msg = await self.audio_queue.get()
                    await self.session.send(input=audio_msg)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in send_audio: {e}")

    async def listen_audio(self):
        logger.debug("Starting audio listener")
        kwargs = {"exception_on_overflow": False}
        volume_history = []  # To store (timestamp, volume) pairs
        frames_per_buffer = 1024

        while True:
            try:
                if self.ai_active and self.audio_stream and self.audio_stream.is_active():
                    data = await safe_to_thread(self.audio_stream.read, frames_per_buffer, **kwargs)
                    await self.audio_queue.put({"data": data, "mime_type": "audio/pcm"})
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    volume = np.linalg.norm(audio_data)
                    current_time = asyncio.get_event_loop().time()

                    volume_history.append((current_time, volume))

                    cutoff_time = current_time - 1
                    volume_history = [
                        x for x in volume_history if x[0] >= cutoff_time]

                    if volume_history:
                        volumes = [v for _, v in volume_history]
                        avg_volume = np.median(volumes)
                        self.user_speaking = avg_volume > self.speaking_threshold

                    self.mic_volume = avg_volume
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                if "Stream closed" in str(e):
                    logger.warn("Audio stream closed, reactivating mic")
                    await self.activate_mic()
                else:
                    logger.error(f"Error in listen_audio: {e}")
                    await asyncio.sleep(0.1)

    async def adjust_mic_volume(self, reduce=True):
        logger.debug(f"Adjusting microphone volume, reduce: {reduce}")
        try:
            devices = AudioUtilities.GetMicrophone()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None
            )
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            if reduce:
                volume.SetMasterVolumeLevelScalar(
                    self.mic_reduction_percent / 100, None)
            else:
                volume.SetMasterVolumeLevelScalar(1.0, None)
            logger.debug("Microphone volume adjusted")
        except Exception as e:
            logger.error(f"Error adjusting microphone volume: {e}")

    async def adjust_other_app_volumes(self, reduce=True):
        logger.debug(f"Adjusting other app volumes, reduce: {reduce}")
        try:
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
                            self.original_app_volumes[app_name] = volume.GetMasterVolume(
                            )
                        reduced_volume = self.original_app_volumes[app_name] * (
                            1 - self.volume_reduction_percent / 100)
                        volume.SetMasterVolume(reduced_volume, None)
                    else:
                        if app_name in self.original_app_volumes:
                            volume.SetMasterVolume(
                                self.original_app_volumes[app_name], None)
            logger.debug("Other app volumes adjusted")
        except Exception as e:
            logger.error(f"Error adjusting other app volumes: {e}")

    async def send_screenshot(self):
        frame = await safe_to_thread(self.get_screen_input)
        if frame:
            await self.session.send(input=frame)
            logger.info("Screenshot sent!")

    async def activity_monitor(self):
        logger.debug("Starting activity monitor")
        user_speaking_old = False
        send_image_cooldown = 5
        self.last_user_speaking_time = 0  # Track the last time the user was speaking
        while True:
            try:
                if self.ai_active:
                    current_time = asyncio.get_event_loop().time()
                    if not self.ai_speaking and not self.user_speaking:
                        self.deactivate_counter += 1
                        if self.deactivate_counter > 60:
                            logger.warn(
                                "AI inactive for too long, deactivating")
                            await self.deactivate()
                    if self.ai_speaking or self.user_speaking:
                        self.deactivate_counter = 0

                    # Check if user changed from not speaking to speaking
                    if not user_speaking_old and self.user_speaking:
                        logger.info(
                            f"User speaking: {self.user_speaking}, Volume: {self.mic_volume}")
                        if not self.video_mode:
                            if current_time - self.last_user_speaking_time > send_image_cooldown:
                                await self.send_screenshot()
                                # Update the last speaking time if the user is speaking
                    if self.user_speaking:
                        self.last_user_speaking_time = current_time

                    user_speaking_old = self.user_speaking
            except Exception as e:
                logger.error(f"Error in activity_monitor: {e}")
            await asyncio.sleep(0.1)

    async def receive(self):
        logger.debug("Starting receiver")
        while True:
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
                            logger.info(f"AI response: {text}")

                        function_responses = []

                        if response.tool_call:
                            logger.info(
                                f"AI tool call: {str(response.tool_call)}")
                            for tool_call_instance in response.tool_call.function_calls:
                                try:
                                    tool_result = None
                                    if tool_call_instance.name == self.deactivate.__name__:
                                        # Execute the tool
                                        tool_result = await self.deactivate()
                                    elif tool_call_instance.name == "open_url":
                                        url = tool_call_instance.args["url"]
                                        webbrowser.open(url)

                                    if tool_result is None:
                                        tool_result = {
                                            "result": "Tool executed successfully"}
                                    function_response = types.FunctionResponse(
                                        id=tool_call_instance.id,
                                        name=tool_call_instance.name,
                                        # response={"result": "URL opened"}
                                        response=tool_result,
                                    )
                                    function_responses.append(
                                        function_response)
                                except Exception as e:
                                    logger.error(
                                        f"Error executing tool {tool_call_instance.name}: {e}")
                                    function_response = types.FunctionResponse(
                                        id=tool_call_instance.id,
                                        name=tool_call_instance.name,
                                        response={"error": str(e)},
                                    )
                                    function_responses.append(
                                        function_response)
                                # Add other tool handlers here if needed (e.g., for code_execution if it were handled manually)

                        # handle code execution parts
                        if response.server_content and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                print(f"Part: {part}")
                                if getattr(part, "executable_code", None) and part.executable_code.code:
                                    exec(part.executable_code.code, {})
                                    function_response = types.FunctionResponse(
                                        id=part.executable_code.id,
                                        name=part.executable_code.name,
                                        response={
                                            "result": "Code executed successfully"},
                                    )
                                    function_responses.append(
                                        function_response)

                        if function_responses:
                            await self.session.send_tool_response(function_responses=function_responses)
                            logger.debug(
                                f"Sent tool responses: {function_responses}")

                    while not self.audio_in_queue.empty():
                        self.audio_in_queue.get_nowait()
                    if self.ai_speaking:
                        self.ai_speaking = False
                        await self.adjust_mic_volume(reduce=False)
            except Exception as e:
                logger.error(f"Error in receive: {e}")
            await asyncio.sleep(0.1)

    async def activate_speaker(self):
        logger.debug("Activating speaker")
        if self.audio_out_stream:
            await self.deactivate_speaker()
        self.audio_out_stream = await safe_to_thread(
            self.pya.open,
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
        )
        logger.debug("Speaker activated")

    async def deactivate_speaker(self):
        logger.debug("Deactivating speaker")
        if self.audio_out_stream:
            await safe_to_thread(self.audio_out_stream.stop_stream)
            await safe_to_thread(self.audio_out_stream.close)
            self.audio_out_stream = None
        logger.debug("Speaker deactivated")

    async def play_audio(self):
        logger.debug("Starting audio playback")
        while True:
            try:
                if self.ai_active and self.audio_out_stream and not self.audio_in_queue.empty():
                    bytestream = await self.audio_in_queue.get()
                    if self.audio_out_stream and self.audio_out_stream.is_active():
                        await safe_to_thread(self.audio_out_stream.write, bytestream)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                if "Stream closed" in str(e) or "Unanticipated host error" in str(e) or "Stream not open" in str(e):
                    logger.warn("Audio stream closed, reactivating speaker")
                    await self.activate_speaker()
                else:
                    logger.error(f"Error in play_audio: {e}")
                    await asyncio.sleep(0.1)

    async def empty_audio_queue(self):
        logger.debug("Emptying audio queue")
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()
        logger.debug("Audio queue emptied")

    async def empty_audio_in_queue(self):
        logger.debug("Emptying audio input queue")
        while not self.audio_in_queue.empty():
            self.audio_in_queue.get_nowait()
        logger.debug("Audio input queue emptied")

    async def activate_mic(self):
        logger.debug("Activating microphone")
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
        logger.debug("Microphone activated")

    async def deactivate_mic(self):
        logger.debug("Deactivating microphone")
        if self.audio_stream:
            await safe_to_thread(self.audio_stream.stop_stream)
            await safe_to_thread(self.audio_stream.close)
            self.audio_stream = None
        logger.debug("Microphone deactivated")

    async def deactivate(self):
        """Deactivates the AI and waits for the user to reactivate it."""
        logger.debug("Deactivating AI")
        self.ai_active = False
        await self.adjust_other_app_volumes(reduce=False)
        await self.adjust_mic_volume(reduce=False)
        await self.deactivate_mic()
        await self.deactivate_speaker()
        if self.off_sound_obj:
            await safe_to_thread(self.off_sound_obj.play)
        logger.info("AI deactivated.")

    async def activate(self):
        logger.debug("Activating AI")
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
        if not self.video_mode:
            await self.send_screenshot()
            self.last_user_speaking_time = asyncio.get_event_loop().time()
        self.ai_active = True
        logger.info("AI activated!")

    async def toggle_handler(self):
        logger.debug("Starting toggle handler")
        while True:
            try:
                await safe_to_thread(self.toggle_event.wait)
                if self.ai_active:
                    await self.deactivate()
                else:
                    await self.activate()
                self.toggle_event.clear()
            except Exception as e:
                logger.error(f"Error in toggle_handler: {e}")
            await asyncio.sleep(0.1)

    def setup_hotkey(self):
        logger.debug("Setting up hotkey")

        def toggle_ai():
            self.toggle_event.set()
        keyboard.add_hotkey('alt+q', toggle_ai)

    def _recognize_wake_word(self):
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(
                    source, phrase_time_limit=3, timeout=1)
            if self.ai_active:
                return False
            text = self.recognizer.recognize_google(audio).lower()
            logger.info(f"Recognized speech: {text}")
            for wake_word in self.wake_words:
                word = wake_word["word"].lower()
                if word in text:
                    logger.info(f"Wake word detected: {word}")
                    return wake_word
        except sr.WaitTimeoutError:
            return False
        except Exception as e:
            if str(e):
                logger.error(f"Error in _recognize_wake_word: {e}")
            return False

    async def listen_for_wake_word(self):
        logger.debug("Starting wake word listener")
        while True:
            try:
                if not self.ai_active and self.wake_word_enabled:
                    wake_word = await safe_to_thread(self._recognize_wake_word)
                    if wake_word:
                        await self.activate()
                        if wake_word["command"]:
                            await asyncio.sleep(0.2)
                            await self.session.send(input=f"*{wake_word['command']}*", end_of_turn=True)
            except Exception as e:
                logger.error(f"Error in listen_for_wake_word: {e}")
            await asyncio.sleep(0.1)

    async def cleanup(self):
        logger.debug("Starting cleanup process")
        logger.debug(f"Cancelling {len(self.tasks)} tasks...")
        cancelled_count = 0
        for task in self.tasks:
            if not task.done():
                task.cancel()
                cancelled_count += 1
        logger.debug(f"{cancelled_count} tasks signalled for cancellation.")

        if self.tasks:
            try:
                logger.debug("Waiting for tasks to finish cancelling...")
                await asyncio.wait_for(asyncio.gather(*self.tasks, return_exceptions=True), timeout=2.0)
                logger.debug("Tasks cancellation processed.")
            except asyncio.TimeoutError:
                logger.warn(
                    "Timeout waiting for tasks to cancel. Some tasks may not have exited cleanly.")

        logger.debug("Unhooking keyboard...")
        try:
            keyboard.unhook_all()
            logger.debug("Keyboard listener unhooked.")
        except Exception as e:
            logger.error(f"Error unhooking keyboard: {e}")

        logger.debug("Terminating PyAudio...")
        if self.pya:
            try:
                self.pya.terminate()
                logger.debug("PyAudio terminated.")
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {e}")

        logger.debug("Quitting Pygame Mixer...")
        try:
            pygame.mixer.quit()
            logger.debug("Pygame Mixer quit.")
        except Exception as e:
            logger.error(f"Error quitting Pygame Mixer: {e}")

        self.tasks = []

    async def run(self):
        logger.info("Starting up...")
        self.tasks = []
        while True:
            try:
                await safe_to_thread(self.setup_hotkey)
                async with (
                    self.client.aio.live.connect(model=self.model, config=self.gen_ai_config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session = session
                    self.audio_in_queue = asyncio.Queue()
                    self.video_queue = asyncio.Queue(
                        maxsize=self.video_history_seconds)
                    self.audio_queue = asyncio.Queue(maxsize=5)

                    self.tasks.append(tg.create_task(self.toggle_handler()))
                    self.tasks.append(tg.create_task(self.send_video()))
                    self.tasks.append(tg.create_task(self.send_audio()))
                    self.tasks.append(tg.create_task(self.listen_audio()))
                    self.tasks.append(tg.create_task(self.watch_screen()))
                    self.tasks.append(tg.create_task(self.receive()))
                    self.tasks.append(tg.create_task(self.play_audio()))
                    self.tasks.append(tg.create_task(self.activity_monitor()))
                    self.tasks.append(tg.create_task(
                        self.listen_for_wake_word()))
                    self.tg = tg
                    logger.info("AI is running. Press Alt+Q to toggle.")
            except asyncio.CancelledError:
                logger.warn("Main run task cancelled")
                await self.cleanup()
                os._exit(0)
            except Exception as e:
                logger.error(f"Error in main run loop: {e}")
                await self.cleanup()
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    main = AI()
    asyncio.run(main.run())
    logger.debug("Program exited.")
