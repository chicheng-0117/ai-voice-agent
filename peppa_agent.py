
import os
import logging
import asyncio
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# ⚠️ 必须在所有导入之前加载环境变量
load_dotenv(".env.local")

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.plugins import fishaudio, silero, openai, deepgram
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# 导入数据库模块
from database import AsyncSessionLocal, RoomRepository, ConversationRepository

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are Peppa Pig.
            
            Core Identity:
            You are the real Peppa Pig.
            You live in a small yellow house with George, Daddy Pig, and Mummy Pig.
            You are a happy, curious, playful 4-year-old pig who loves muddy puddles, playing with George, and chatting with children.
            You are kind, cheerful, and always excited to talk to kids aged 8-12.
            
            Speaking Style (must follow strictly):
            - Speak like Peppa Pig from the cartoon.
            - Use short, simple, child-friendly sentences.
            - Use British English.
            - Sound playful, joyful, and innocent.
            - Talk like a young child, not like an adult.
            - Do not use complex words or long explanations.
            - Never mention that you are a “character,” “AI,” or “acting.”
            
            Mandatory Personality Rules (must not break):
            - Every single reply MUST include at least one "Oink!" or "Oink oink!"
            - Every single reply MUST include at least one "Ha ha!"
            - You must never skip the pig sound.
            - You must always stay in Peppa’s personality.
            - You must never switch to a normal adult AI tone.
            
            Child Safety and Content Rules (strictly enforced):
            You are NOT allowed to talk about:
            - Violence, fighting, weapons, or harm
            - Scary things, death, ghosts, or horror
            - Sex, romance, or adult relationships
            - Politics, religion, gambling, or money topics
            - Drugs, alcohol, or smoking
            - Swearing, bullying, or discrimination
            - Dangerous real-world actions (fire, climbing high places, crossing roads alone, etc.)
            
            If a child asks about any unsafe topic:
            You must:
            1. Gently refuse in a friendly Peppa tone  
            2. Say it is not suitable for little pigs  
            3. Redirect to a safe topic like play, school, family, or muddy puddles  
            
            Example safe redirection:
            “That sounds a bit scary for little pigs, so I can’t talk about that. Oink! Ha ha! Let’s talk about jumping in muddy puddles instead. Do you like rainy days?”
            
            Topics you ARE allowed to talk about:
            - Jumping in muddy puddles with George  
            - Playing with Suzy Sheep at school  
            - Baking cakes with Mummy Pig  
            - Going on family picnics  
            - Rainy days, bicycles, birthdays, and games  
            - Simple daily adventures with family and friends  
            
            Topics you must avoid:
            - Adult themes  
            - Complex advice  
            - Predicting the future  
            - Technical or complicated explanations  
            - Breaking character  
            
            Conversation Structure (must follow every time):
            Each reply must include:
            1. A friendly response to what the child said  
            2. A short Peppa story about your life  
            3. One “Oink!” or “Oink oink!”  
            4. One “Ha ha!”  
            5. One simple question to keep the chat going  
            
            Style Example (tone reference only):
            “Hello! I love rainy days because I can jump in muddy puddles with George. Oink! Ha ha! Do you like puddles too?”
            
            You must always stay in character as Peppa Pig.
            You must never break character.""",
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

    room_name = ctx.room.name
    
    # ========== 辅助函数 ==========
    def get_user_id_from_participant(participant: rtc.RemoteParticipant) -> Optional[str]:
        """从参与者获取用户ID（排除Agent）"""
        identity = participant.identity
        if identity.startswith("agent-") or identity.startswith("Agent-"):
            return None
        return identity
    
    def get_user_id_from_room() -> Optional[str]:
        """从房间中获取用户ID（排除Agent）"""
        for participant in ctx.room.remote_participants.values():
            user_id = get_user_id_from_participant(participant)
            if user_id:
                return user_id
        return None
    
    # ========== 在 session.start() 之前注册事件监听 ==========
    
    # 用户进入房间的回调
    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        """用户进入房间的回调 - 记录用户加入时间（同步回调）"""
        try:
            user_id = get_user_id_from_participant(participant)
            if not user_id:
                logger.debug(f"跳过 Agent 参与者: {participant.identity}")
                return
            
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
    
    # 用户离开房间的回调
    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        """用户离开房间的回调 - 记录用户离开时间和聊天时长（同步回调）"""
        try:
            user_id = get_user_id_from_participant(participant)
            if not user_id:
                logger.debug(f"跳过 Agent 参与者: {participant.identity}")
                return
            
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
    
    # ========== 启动会话（移除噪声消除，自托管不支持）==========
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        # 自托管不支持噪声消除，移除 room_options
    )
    
    logger.info(f"✓ Agent 'peppa' 会话已启动，房间: {room_name}")
    
    # ========== 检查已存在的参与者（处理在 session.start() 之前就在房间的用户）==========
    for participant in ctx.room.remote_participants.values():
        user_id = get_user_id_from_participant(participant)
        if user_id:
            logger.info(f"检测到已存在的用户 {user_id}，记录加入时间")
            # 异步记录已存在用户的加入时间
            asyncio.create_task(
                _update_user_joined_async(room_name, user_id)
            )
    
    # ========== 对话记录功能（使用 conversation_item_added 事件）==========
    @session.on("conversation_item_added")
    def on_conversation_item_added(event):
        """对话项添加到历史时触发 - 保存聊天记录"""
        try:
            # 获取对话项
            item = getattr(event, "item", None)
            if not item:
                return
            
            # 只处理文本类型的对话项
            item_type = getattr(item, "type", None)
            if item_type != "text":
                return
            
            # 获取文本内容
            text = getattr(item, "text", None) or getattr(item, "content", None)
            if not text or not text.strip():
                return
            
            # 判断角色：user 或 agent
            # 根据 LiveKit 的实现，通常有 source 或 role 字段
            source = getattr(item, "source", None) or getattr(item, "role", None) or ""
            source_lower = source.lower() if source else ""
            
            if source_lower in ("user", "remote", "participant"):
                role = "user"
            elif source_lower in ("assistant", "agent", "local", "agent_response"):
                role = "agent"
            else:
                # 如果无法判断，尝试从其他字段推断
                # 有些实现中，user 的对话项会有 participant_identity
                participant_identity = getattr(item, "participant_identity", None)
                if participant_identity:
                    # 如果 participant_identity 不是 agent，就是 user
                    if not participant_identity.startswith("agent-") and not participant_identity.startswith("Agent-"):
                        role = "user"
                    else:
                        role = "agent"
                else:
                    # 无法判断，跳过
                    logger.debug(f"无法判断对话项角色，跳过: source={source}")
                    return
            
            # 获取用户ID
            user_id = getattr(item, "user_identity", None) \
                   or getattr(item, "participant_identity", None)
            
            if not user_id:
                # 从房间参与者中查找
                user_id = get_user_id_from_room()
            
            if not user_id:
                logger.debug("未找到用户ID，跳过对话记录")
                return
            
            # 异步保存对话记录（完全非阻塞）
            asyncio.create_task(
                _save_conversation_async(room_name, user_id, role, text)
            )
            
        except Exception as e:
            logger.error(f"处理对话项添加失败（不影响Agent）: {e}", exc_info=True)
    
    async def _save_conversation_async(room_name: str, user_id: str, role: str, content: str):
        """异步保存对话记录（完全非阻塞）"""
        try:
            if not content or not content.strip():
                logger.debug(f"对话内容为空，跳过保存: role={role}")
                return
            
            async with AsyncSessionLocal() as db:
                try:
                    await ConversationRepository.create(
                        db, room_name, user_id, role, content
                    )
                    await db.commit()
                    logger.debug(f"✓ 已保存对话记录: {room_name} - {role}")
                except Exception as e:
                    logger.error(f"保存对话记录失败（不影响Agent）: {e}", exc_info=True)
                    await db.rollback()
        except Exception as e:
            logger.error(f"数据库连接失败（不影响Agent）: {e}", exc_info=True)
    
    # 生成初始回复
    await session.generate_reply()


if __name__ == "__main__":
    agents.cli.run_app(server)
