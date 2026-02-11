
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
    
    # 如果房间名称是 "console"，跳过 agent:peppa 校验
    room_name = ctx.room.name
    is_console_room = room_name.lower() == "console"
    
    if not is_console_room:
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
    else:
        logger.info(
            f"✓ Agent 'peppa' 处理 console 房间 {ctx.room.name}（跳过 metadata 校验）"
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
    
    # ========== 对话记录功能（使用多种方式确保能获取到）==========
    
    async def _save_conversation_async(room_name: str, user_id: str, role: str, content: str):
        """异步保存对话记录（完全非阻塞）"""
        try:
            logger.debug(f"开始执行 _save_conversation_async: room={room_name}, user={user_id}, role={role}")
            
            if not content or not content.strip():
                logger.debug(f"对话内容为空，跳过保存: role={role}, room={room_name}")
                return
            
            logger.info(f"DEBUG: 准备保存对话记录: room={room_name}, user={user_id}, role={role}, content={content[:50]}...")
            
            logger.debug(f"准备连接数据库: room={room_name}")
            async with AsyncSessionLocal() as db:
                try:
                    logger.debug(f"调用 ConversationRepository.create: room={room_name}, user={user_id}, role={role}")
                    await ConversationRepository.create(
                        db, room_name, user_id, role, content
                    )
                    logger.debug(f"准备提交事务: room={room_name}")
                    await db.commit()
                    logger.info(f"✓ 已保存对话记录: {room_name} - {role} - {content[:50]}...")
                except Exception as e:
                    logger.error(f"保存对话记录失败（不影响Agent）: room={room_name}, error={e}", exc_info=True)
                    await db.rollback()
        except Exception as e:
            logger.error(f"数据库连接失败（不影响Agent）: {e}", exc_info=True)
    
    def _handle_conversation_event(event, event_name: str = ""):
        """处理对话事件（通用处理函数）"""
        try:
            logger.info(f"DEBUG: 对话事件触发: event_name={event_name}, event={event}, type={type(event)}")
            
            # 尝试多种方式获取内容
            content = None
            role = None
            
            # 优先处理 ConversationItemAddedEvent（有 item 属性）
            if hasattr(event, "item"):
                item = event.item
                logger.info(f"DEBUG: 从 event.item 获取: item={item}, item_type={type(item)}")
                
                # 获取 content（可能是列表或字符串）
                raw_content = getattr(item, "content", None)
                if raw_content:
                    # 如果是列表，合并为字符串
                    if isinstance(raw_content, list):
                        content = " ".join(str(c) for c in raw_content if c)
                    elif isinstance(raw_content, str):
                        content = raw_content
                    else:
                        content = str(raw_content)
                else:
                    # 尝试其他字段
                    raw_content = getattr(item, "text", None) or getattr(item, "message", None)
                    if raw_content:
                        if isinstance(raw_content, list):
                            content = " ".join(str(c) for c in raw_content if c)
                        else:
                            content = str(raw_content)
                
                # 获取角色（直接从 item.role 获取）
                role = getattr(item, "role", None) or getattr(item, "source", None) or getattr(item, "sender", None)
                
                logger.info(f"DEBUG: 提取结果: content={content}, role={role}")
            
            # 如果 event 本身就是字符串
            elif isinstance(event, str):
                content = event
                role = "user"  # 默认假设是用户消息
            # 如果 event 有 content 或 text 属性
            elif hasattr(event, "content"):
                raw_content = event.content
                if isinstance(raw_content, list):
                    content = " ".join(str(c) for c in raw_content if c)
                else:
                    content = str(raw_content) if raw_content else None
                role = getattr(event, "role", None) or getattr(event, "source", None)
            elif hasattr(event, "text"):
                raw_content = event.text
                if isinstance(raw_content, list):
                    content = " ".join(str(c) for c in raw_content if c)
                else:
                    content = str(raw_content) if raw_content else None
                role = getattr(event, "role", None) or getattr(event, "source", None)
            elif hasattr(event, "message"):
                raw_content = event.message
                if isinstance(raw_content, list):
                    content = " ".join(str(c) for c in raw_content if c)
                else:
                    content = str(raw_content) if raw_content else None
                role = getattr(event, "role", None) or getattr(event, "source", None)
            # 如果是字典类型
            elif isinstance(event, dict):
                raw_content = event.get("content") or event.get("text") or event.get("message")
                if raw_content:
                    if isinstance(raw_content, list):
                        content = " ".join(str(c) for c in raw_content if c)
                    else:
                        content = str(raw_content)
                role = event.get("role") or event.get("source") or event.get("sender")
            
            # 验证内容（处理字符串和列表）
            if not content:
                logger.debug(f"事件内容为空，跳过: event_name={event_name}")
                return
            
            # 如果是字符串，检查是否为空
            if isinstance(content, str):
                if not content.strip():
                    logger.debug(f"事件内容为空字符串，跳过: event_name={event_name}")
                    return
            # 如果是其他类型，转换为字符串
            else:
                content = str(content)
                if not content.strip():
                    logger.debug(f"事件内容转换后为空，跳过: event_name={event_name}")
                    return
            
            # 判断角色
            if not role:
                # 从事件名称推断
                if "user" in event_name.lower():
                    role = "user"
                elif "agent" in event_name.lower() or "assistant" in event_name.lower():
                    role = "agent"
                else:
                    # 尝试从 participant_identity 推断
                    participant_identity = None
                    if hasattr(event, "participant_identity"):
                        participant_identity = event.participant_identity
                    elif hasattr(event, "user_identity"):
                        participant_identity = event.user_identity
                    elif isinstance(event, dict):
                        participant_identity = event.get("participant_identity") or event.get("user_identity")
                    
                    if participant_identity:
                        role = "user" if not participant_identity.startswith("agent-") else "agent"
                    else:
                        role = "user"  # 默认假设是用户消息
            
            # 标准化角色名称
            if role.lower() in ("user", "remote", "participant"):
                role = "user"
            elif role.lower() in ("assistant", "agent", "local"):
                role = "agent"
            
            # 获取用户ID
            user_id = None
            
            # 优先从 event.item 获取（如果是 ConversationItemAddedEvent）
            if hasattr(event, "item"):
                item = event.item
                user_id = getattr(item, "user_identity", None) \
                       or getattr(item, "participant_identity", None) \
                       or getattr(item, "identity", None)
            
            # 如果还没有，从 event 本身获取
            if not user_id:
                if hasattr(event, "user_identity"):
                    user_id = event.user_identity
                elif hasattr(event, "participant_identity"):
                    user_id = event.participant_identity
                elif isinstance(event, dict):
                    user_id = event.get("user_identity") or event.get("participant_identity")
            
            # 最后从房间参与者中查找
            if not user_id:
                user_id = get_user_id_from_room()
            
            if not user_id:
                logger.warning(f"未找到用户ID，跳过对话记录: event_name={event_name}, room={room_name}")
                return
            
            logger.info(f"DEBUG: 处理对话事件: role={role}, user_id={user_id}, content={content[:50]}...")
            
            # 异步保存对话记录（完全非阻塞）
            logger.debug(f"创建异步任务保存对话记录: room={room_name}, user={user_id}, role={role}")
            asyncio.create_task(
                _save_conversation_async(room_name, user_id, role, content)
            )
            logger.debug(f"异步任务已创建: room={room_name}")
            
        except Exception as e:
            logger.error(f"处理对话事件失败（不影响Agent）: {e}", exc_info=True)
    
    def setup_conversation_hooks():
        """设置对话记录钩子（尝试多种可能的事件）"""
        logger.info(f"开始设置对话记录钩子，房间: {room_name}")
        
        # 尝试注册 conversation_item_added 事件
        try:
            @session.on("conversation_item_added")
            def on_conversation_item_added(event):
                logger.debug(f"DEBUG: conversation_item_added 事件被触发")
                _handle_conversation_event(event, "conversation_item_added")
            logger.info("✓ 成功注册 conversation_item_added 事件")
        except Exception as e:
            logger.warning(f"无法注册 conversation_item_added 事件: {e}")
        
        # 尝试注册 user_message 事件
        try:
            @session.on("user_message")
            def on_user_message(event):
                _handle_conversation_event(event, "user_message")
            logger.info("✓ 成功注册 user_message 事件")
        except Exception as e:
            logger.debug(f"无法注册 user_message 事件: {e}")
        
        # 尝试注册 agent_message 事件
        try:
            @session.on("agent_message")
            def on_agent_message(event):
                _handle_conversation_event(event, "agent_message")
            logger.info("✓ 成功注册 agent_message 事件")
        except Exception as e:
            logger.debug(f"无法注册 agent_message 事件: {e}")
        
        # 尝试注册 message_added 事件
        try:
            @session.on("message_added")
            def on_message_added(event):
                _handle_conversation_event(event, "message_added")
            logger.info("✓ 成功注册 message_added 事件")
        except Exception as e:
            logger.debug(f"无法注册 message_added 事件: {e}")
    
    async def save_conversation_periodically():
        """定期检查并保存对话历史（最可靠的方式）"""
        saved_message_hashes = set()
        
        while True:
            try:
                # 直接访问 session.conversation
                if hasattr(session, "conversation"):
                    conv = session.conversation
                    messages = []
                    
                    # 尝试多种方式获取消息列表
                    if hasattr(conv, "messages"):
                        try:
                            messages = list(conv.messages) if hasattr(conv.messages, "__iter__") else []
                        except:
                            messages = []
                    elif hasattr(conv, "items"):
                        try:
                            messages = list(conv.items) if hasattr(conv.items, "__iter__") else []
                        except:
                            messages = []
                    elif hasattr(conv, "history"):
                        try:
                            messages = list(conv.history) if hasattr(conv.history, "__iter__") else []
                        except:
                            messages = []
                    elif hasattr(conv, "__iter__"):
                        try:
                            messages = list(conv)
                        except:
                            messages = []
                    
                    logger.debug(f"DEBUG: 对话历史中有 {len(messages)} 条消息")
                    
                    # 处理每条消息
                    for idx, msg in enumerate(messages):
                        try:
                            # 获取消息内容（可能是列表或字符串）
                            content = None
                            raw_content = None
                            
                            if hasattr(msg, "content"):
                                raw_content = msg.content
                            elif hasattr(msg, "text"):
                                raw_content = msg.text
                            elif hasattr(msg, "message"):
                                raw_content = msg.message
                            elif isinstance(msg, str):
                                raw_content = msg
                            
                            # 处理列表类型的 content
                            if raw_content:
                                if isinstance(raw_content, list):
                                    content = " ".join(str(c) for c in raw_content if c)
                                elif isinstance(raw_content, str):
                                    content = raw_content
                                else:
                                    content = str(raw_content)
                            
                            if not content or (isinstance(content, str) and not content.strip()):
                                continue
                            
                            # 创建唯一标识符（使用索引+内容前50字符）
                            msg_hash = hash(f"{idx}_{content[:50]}")
                            if msg_hash in saved_message_hashes:
                                continue
                            
                            saved_message_hashes.add(msg_hash)
                            
                            # 获取角色
                            role = None
                            if hasattr(msg, "role"):
                                role = msg.role
                            elif hasattr(msg, "source"):
                                role = msg.source
                            elif hasattr(msg, "sender"):
                                role = msg.sender
                            
                            # 判断角色
                            if role:
                                role_lower = role.lower()
                                if role_lower in ("user", "remote", "participant"):
                                    role = "user"
                                elif role_lower in ("assistant", "agent", "local"):
                                    role = "agent"
                                else:
                                    # 尝试从 participant_identity 推断
                                    participant_identity = getattr(msg, "participant_identity", None)
                                    if participant_identity:
                                        role = "user" if not participant_identity.startswith("agent-") else "agent"
                                    else:
                                        # 根据索引推断（通常第一条是用户，第二条是agent，以此类推）
                                        role = "user" if idx % 2 == 0 else "agent"
                            else:
                                # 无法判断角色，根据索引推断
                                role = "user" if idx % 2 == 0 else "agent"
                            
                            # 获取用户ID
                            user_id = getattr(msg, "user_identity", None) \
                                   or getattr(msg, "participant_identity", None)
                            
                            if not user_id:
                                user_id = get_user_id_from_room()
                            
                            if user_id:
                                logger.info(f"DEBUG: 从对话历史保存记录: idx={idx}, role={role}, content={content[:50]}...")
                                asyncio.create_task(
                                    _save_conversation_async(room_name, user_id, role, content)
                                )
                                
                        except Exception as e:
                            logger.error(f"处理单条消息失败: {e}", exc_info=True)
                
                await asyncio.sleep(2)  # 每2秒检查一次
                
            except Exception as e:
                logger.error(f"监控对话历史失败（不影响Agent）: {e}", exc_info=True)
                await asyncio.sleep(5)  # 出错后等待更长时间
    
    # 设置事件钩子
    logger.info(f"准备设置对话记录功能，房间: {room_name}")
    setup_conversation_hooks()
    logger.info(f"✓ 对话记录钩子设置完成，房间: {room_name}")
    
    # 启动后台任务定期检查对话历史
    logger.info(f"启动定期检查对话历史任务，房间: {room_name}")
    asyncio.create_task(save_conversation_periodically())
    logger.info(f"✓ 定期检查对话历史任务已启动，房间: {room_name}")
    
    # 生成初始回复
    await session.generate_reply()


if __name__ == "__main__":
    agents.cli.run_app(server)
