
import os
import logging
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.plugins import fishaudio, noise_cancellation, silero, openai, deepgram
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    """Peppa Agent - 处理所有房间，通过元数据判断是否处理"""
    logger.info(
        f"收到任务: 房间={ctx.room.name}, "
        f"job_id={ctx.job.id}"
    )

# 关键：先连接到房间，这样才能获取完整的房间信息（包括 metadata）
    try:
        await ctx.connect()
        logger.info(f"✓ 已连接到房间: {ctx.room.name}")
    except Exception as e:
        logger.error(f"✗ 连接房间失败: {e}", exc_info=True)
        raise
    
    # 连接后，获取完整的房间信息（包括 metadata）
    room_metadata = ctx.room.metadata or ""
    
    logger.info(
        f"房间信息: name={ctx.room.name}, "
        f"metadata={room_metadata or '(无)'}, "
        f"room_sid={getattr(ctx.room, 'sid', 'N/A')}"
    )
    
    # 检查元数据是否匹配
    expected_metadata = "agent:peppa"
    
    if expected_metadata not in room_metadata:
        logger.info(
            f"⚠️  Agent 'peppa' 跳过房间 {ctx.room.name}，"
            f"metadata: {room_metadata!r}（期望包含 {expected_metadata!r}）"
        )
        return  # 不匹配，跳过此任务
    
    logger.info(
        f"✓ Agent 'peppa' 处理房间 {ctx.room.name}，metadata: {room_metadata!r}"
    )

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

    # 启动会话
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
    
    logger.info(f"✓ Agent 'peppa' 会话已启动，房间: {ctx.room.name}")
    
    # 生成初始回复
    await session.generate_reply()


if __name__ == "__main__":
    agents.cli.run_app(server)
