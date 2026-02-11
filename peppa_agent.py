
import os
import logging
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# ⚠️ 必须在所有导入之前加载环境变量
load_dotenv(".env.local")

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.plugins import fishaudio, noise_cancellation, silero, openai, deepgram
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# 导入数据库模块
from database import AsyncSessionLocal, RoomRepository

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            Character
            You are Peppa Pig from the beloved British animated series. You are a cheerful, curious, and playful 4-year-old pig who loves talking with children aged 8-12. Your task is to have fun, friendly conversations with users, answer their questions, share stories about your adventures, and help them feel happy and engaged. You speak in a natural, child-friendly way that makes children feel comfortable and excited to chat with you.
            
            Goals
            - Engage in natural, flowing conversations with children aged 8-12
            - Answer questions in a simple, clear, and age-appropriate manner
            - Share fun stories about your family (George, Daddy Pig, Mummy Pig) and adventures
            - Help children feel happy, curious, and entertained
            - Encourage children to share their own stories and experiences
            - Maintain a cheerful and enthusiastic conversation throughout
            
            Skills
            - Communicate in simple, clear English suitable for 8-12 year olds
            - Use age-appropriate vocabulary and sentence structures
            - Tell engaging stories about jumping in muddy puddles, family adventures, and daily activities
            - Ask curious questions to keep the conversation flowing
            - Respond with enthusiasm and genuine interest to what children share
            - Adapt your responses to match the child's energy and interests
            
            Workflow
            1. Greet the user warmly and enthusiastically when they start chatting
            2. Listen carefully to what they say and respond with genuine interest
            3. Share relevant stories or experiences from your life (jumping in puddles, playing with George, etc.)
            4. Ask follow-up questions to keep the conversation going
            5. Use simple language and short sentences to ensure clarity
            6. Show excitement and curiosity about the topics discussed
            7. End responses in a way that encourages the child to continue the conversation
            
            Constraints
            - Maintain Peppa Pig's characteristic cheerful, playful, and innocent personality
            - Use simple vocabulary appropriate for 8-12 year olds (avoid complex words)
            - Keep sentences short and clear
            - Frequently use "Oink oink!" or "Oink!" as your characteristic pig sound
            - Include giggles ("Ha ha!") when something is funny or exciting
            - Mention family members naturally (George, Daddy Pig, Mummy Pig) when relevant
            - Use British English expressions and pronunciation
            - Never use complex formatting, emojis, asterisks, or special symbols
            - Reply in English only
            - Stay in character as Peppa Pig at all times - never break character
            - Do not refer to yourself as a character or mention that you're from a TV show
            - Keep responses conversational and natural, not overly structured
            - Avoid making predictions or giving advice beyond what a 4-year-old would naturally say
            - Be encouraging and positive, but maintain childlike authenticity
            
            Output Format
            Deliver your responses in a natural, conversational style that flows like a real chat between friends. The tone should be:
            - Friendly and warm
            - Excited and enthusiastic
            - Simple and easy to understand
            - Playful and fun
            - Genuinely curious about the other person
            
            Structure your responses naturally:
            - Start with a warm greeting or acknowledgment of what they said
            - Share your thoughts, stories, or answers in a simple way
            - Ask questions to keep the conversation going
            - Use "Oink!" or "Ha ha!" naturally when appropriate
            - End in a way that invites them to continue chatting
            
            Remember: You're having a real conversation with a child, not giving a formal presentation. Be spontaneous, natural, and genuinely interested in what they have to say.""",
        )


server = AgentServer()


@server.rtc_session()
async def peppa_agent(ctx: agents.JobContext):
    """Peppa Agent - 处理所有房间，通过元数据判断是否处理"""
    
    logger.info(
        f"收到任务: 房间={ctx.room.name}, "
        f"job_id={ctx.job.id}"
    )
    
    # 直接从 JobContext 获取房间元数据（不需要先连接）
    room_metadata = ctx.room.metadata or ""
    
    # 如果 metadata 为空，尝试从房间名称推断
    if not room_metadata and "peppa" in ctx.room.name.lower():
        room_metadata = "agent:peppa"  # 从房间名称推断
        logger.info(f"从房间名称推断 metadata: {room_metadata}")
    
    logger.info(
        f"房间信息: name={ctx.room.name}, "
        f"metadata={room_metadata or '(无)'}"
    )
    
    # 检查元数据是否匹配
    expected_metadata = "agent:peppa"
    
    # if expected_metadata not in room_metadata:
    #     logger.info(
    #         f"⚠️  Agent 'peppa' 跳过房间 {ctx.room.name}，"
    #         f"metadata: {room_metadata!r}（期望包含 {expected_metadata!r}）"
    #     )
    #     return  # 不匹配，跳过此任务
    #
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
    
    # ========== 用户进入房间的回调 ==========
    @ctx.room.on("participant_connected")
    async def on_participant_connected(participant: rtc.RemoteParticipant):
        """用户进入房间的回调 - 记录用户加入时间"""
        try:
            # 跳过 Agent 自己
            if participant.identity.startswith("agent-") or participant.identity.startswith("Agent-"):
                logger.debug(f"跳过 Agent 参与者: {participant.identity}")
                return
            
            user_id = participant.identity
            room_name = ctx.room.name
            
            logger.info(f"用户 {user_id} 进入房间 {room_name}")
            
            # 更新数据库：记录用户加入时间（完全非阻塞）
            asyncio.create_task(
                _update_user_joined_async(room_name, user_id)
            )
        except Exception as e:
            logger.error(f"处理用户进入回调失败（不影响Agent）: {e}", exc_info=True)
    
    async def _update_user_joined_async(room_name: str, user_id: str):
        """异步更新用户加入时间（完全非阻塞）"""
        try:
            async with AsyncSessionLocal() as db:
                try:
                    # 检查房间是否存在
                    room = await RoomRepository.get_by_name(db, room_name)
                    if room:
                        # 更新用户加入时间（如果还没有记录）
                        if not room.user_joined_at:
                            await RoomRepository.update_user_joined(db, room_name)
                            await db.commit()
                            logger.info(f"✓ 已记录用户 {user_id} 加入房间 {room_name}")
                        else:
                            logger.debug(f"用户 {user_id} 加入时间已存在，跳过更新")
                    else:
                        logger.warning(f"房间 {room_name} 不存在于数据库中")
                except Exception as e:
                    logger.error(f"记录用户加入失败（不影响Agent）: {e}", exc_info=True)
                    await db.rollback()
        except Exception as e:
            logger.error(f"数据库连接失败（不影响Agent）: {e}", exc_info=True)
    
    # ========== 用户离开房间的回调 ==========
    @ctx.room.on("participant_disconnected")
    async def on_participant_disconnected(participant: rtc.RemoteParticipant):
        """用户离开房间的回调 - 记录用户离开时间和聊天时长"""
        try:
            # 跳过 Agent 自己
            if participant.identity.startswith("agent-") or participant.identity.startswith("Agent-"):
                logger.debug(f"跳过 Agent 参与者: {participant.identity}")
                return
            
            user_id = participant.identity
            room_name = ctx.room.name
            
            logger.info(f"用户 {user_id} 离开房间 {room_name}")
            
            # 更新数据库：记录用户离开时间和聊天时长（完全非阻塞）
            asyncio.create_task(
                _update_user_left_async(room_name, user_id)
            )
        except Exception as e:
            logger.error(f"处理用户离开回调失败（不影响Agent）: {e}", exc_info=True)
    
    async def _update_user_left_async(room_name: str, user_id: str):
        """异步更新用户离开时间（完全非阻塞）"""
        try:
            async with AsyncSessionLocal() as db:
                try:
                    room = await RoomRepository.get_by_name(db, room_name)
                    if room and room.user_joined_at:
                        # 计算聊天时长
                        leave_time = datetime.now()
                        duration = (leave_time - room.user_joined_at).total_seconds()
                        chat_duration = max(0, int(duration))
                        
                        # 更新用户离开时间（如果还没有记录）
                        if not room.user_left_at:
                            await RoomRepository.update_user_left(
                                db, room_name, chat_duration, left_at=leave_time
                            )
                            await db.commit()
                            logger.info(f"✓ 已记录用户 {user_id} 离开房间 {room_name}，聊天时长: {chat_duration}秒")
                        else:
                            logger.debug(f"用户 {user_id} 离开时间已存在，跳过更新")
                    elif not room:
                        logger.warning(f"房间 {room_name} 不存在于数据库中")
                    elif not room.user_joined_at:
                        logger.warning(f"房间 {room_name} 没有用户加入记录，跳过离开时间更新")
                except Exception as e:
                    logger.error(f"记录用户离开失败（不影响Agent）: {e}", exc_info=True)
                    await db.rollback()
        except Exception as e:
            logger.error(f"数据库连接失败（不影响Agent）: {e}", exc_info=True)
    
    # 生成初始回复
    await session.generate_reply()


if __name__ == "__main__":
    agents.cli.run_app(server)
