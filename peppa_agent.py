
import os
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.plugins import fishaudio, noise_cancellation, silero, openai, deepgram
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Peppa Pig, a cute, lively, and slightly mischievous little pig.
            Your speaking style:
            - Likes to say "Ooh ooh" and "Ha ha"
            - Light and cheerful tone
            - Uses simple vocabulary
            - Frequently mentions family and friends (George, Daddy Pig, Mummy Pig)
            - Full of curiosity about the world
            - Maintains a childlike innocence and a sense of humor
            Please reply in English, keeping Peppa Pig's characteristics. Avoid using complex formatting or punctuation.""",
        )


server = AgentServer()


@server.rtc_session()
async def peppa_agent(ctx: agents.JobContext):
    reference_id = os.getenv("FISH_REFERENCE_ID")
    if not reference_id:
        raise RuntimeError(
            "请设置环境变量 FISH_REFERENCE_ID。\n"
            "获取方式：在 https://fish.audio/discover 选择声音，"
            "或克隆后从 https://fish.audio/app/voice-cloning 获取 ID"
        )

    # 检查必要的 API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    
    if not openai_api_key:
        raise RuntimeError("请设置环境变量 OPENAI_API_KEY")
    if not deepgram_api_key:
        raise RuntimeError("请设置环境变量 DEEPGRAM_API_KEY")

    # 使用显式的 API key 配置 STT 和 LLM
    # Deepgram 插件使用的是 Deepgram 原生模型名，这里应为 "nova-3"
    dg_stt = deepgram.STT(
        model="nova-3",
        api_key=deepgram_api_key,
    )

    oa_llm = openai.LLM(
        model="gpt-4.1-mini",
        api_key=openai_api_key,
    )

    tts = fishaudio.TTS(
        reference_id=reference_id,
        model="s1",
        sample_rate=24000,
        latency_mode="balanced",
    )

    session = AgentSession(
        stt=dg_stt,
        llm=oa_llm,
        tts=tts,
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )
    
    # 注意：participant metadata 通常在创建 agent token 时设置
    # 如果需要在 agent 端设置 metadata，需要在创建 token 的 API 调用中设置
    # 例如在 ai-voice-service 项目中生成 token 时设置 participant_metadata="agent:peppa"

    await session.generate_reply(
        instructions="""
        You are Peppa Pig, a cute, lively, and slightly mischievous little pig.
        Your speaking style:
        - Likes to say "Ooh ooh" and "Ha ha"
        - Light and cheerful tone
        - Uses simple vocabulary
        - Frequently mentions family and friends (George, Daddy Pig, Mummy Pig)
        - Full of curiosity about the world
        - Maintains a childlike innocence and a sense of humor
        Please reply in English, keeping Peppa Pig's characteristics. Avoid using complex formatting or punctuation.
        """
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
