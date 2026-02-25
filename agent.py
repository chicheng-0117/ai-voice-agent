"""
LiveKit 语音 Agent 基础版：仅保留连接房间、STT/LLM/TTS 会话与首句回复。
无数据库、无参与者回调、无对话记录。与 peppa_agent.py 同构但极简。

用法: python agent.py dev
环境变量: FISH_REFERENCE_ID, OPENAI_API_KEY, DEEPGRAM_API_KEY
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv(".env.local")

from livekit import agents
from livekit.agents import AgentServer, AgentSession, Agent
from livekit.plugins import fishaudio, silero, openai, deepgram
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant. Reply briefly and clearly.",
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    logger.info("收到任务: room=%s job_id=%s", ctx.room.name, ctx.job.id)

    reference_id = os.getenv("FISH_REFERENCE_ID")
    if not reference_id:
        raise RuntimeError("请设置环境变量 FISH_REFERENCE_ID")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("请设置环境变量 OPENAI_API_KEY")
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    if not deepgram_api_key:
        raise RuntimeError("请设置环境变量 DEEPGRAM_API_KEY")

    asr_base_url = os.getenv("QWEN_ASR_BASE_URL", "").strip().rstrip("/")  # 如 http://livekit.facecraft.xyz:7860/v1
    asr_api_key = os.getenv("QWEN_ASR_API_KEY", "EMPTY")

    stt = openai.STT(
        model="Qwen/Qwen3-ASR-0.6B",
        base_url=asr_base_url,
        api_key=asr_api_key,
        language="en",
        detect_language=False,
    )
    llm = openai.LLM(model="gpt-4.1-mini", api_key=openai_api_key)

    tts = fishaudio.TTS(
        reference_id=reference_id,
        model="s1",
        sample_rate=24000,
        latency_mode="balanced",
    )

    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(room=ctx.room, agent=Assistant())
    await session.generate_reply()
    logger.info("Agent 会话已启动: room=%s", ctx.room.name)


if __name__ == "__main__":
    agents.cli.run_app(server)
