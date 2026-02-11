"""数据库仓库层"""
from typing import Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from database.models import Agent, Room, Conversation
import logging

logger = logging.getLogger(__name__)


class AgentRepository:
    """Agent 数据仓库"""
    
    @staticmethod
    async def get_by_name(session: AsyncSession, agent_name: str) -> Optional[Agent]:
        """根据名称获取 Agent"""
        result = await session.execute(
            select(Agent).where(Agent.agent_name == agent_name)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_all(session: AsyncSession) -> list[Agent]:
        """获取所有 Agent"""
        result = await session.execute(select(Agent))
        return list(result.scalars().all())


class RoomRepository:
    """房间数据仓库"""
    
    @staticmethod
    async def get_by_name(session: AsyncSession, room_name: str) -> Optional[Room]:
        """根据房间名称获取房间"""
        result = await session.execute(
            select(Room).where(Room.room_name == room_name)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def update_user_joined(session: AsyncSession, room_name: str) -> bool:
        """更新用户加入时间"""
        result = await session.execute(
            update(Room)
            .where(Room.room_name == room_name)
            .values(user_joined_at=datetime.now())
        )
        await session.flush()
        return result.rowcount > 0
    
    @staticmethod
    async def update_user_left(
        session: AsyncSession,
        room_name: str,
        chat_duration: int,
        left_at: Optional[datetime] = None
    ) -> bool:
        """更新用户离开时间和聊天时长
        
        Args:
            session: 数据库会话
            room_name: 房间名称
            chat_duration: 聊天时长（秒）
            left_at: 离开时间，如果为 None 则使用当前时间
        """
        if left_at is None:
            left_at = datetime.now()
        
        result = await session.execute(
            update(Room)
            .where(Room.room_name == room_name)
            .values(
                user_left_at=left_at,
                chat_duration=chat_duration,
            )
        )
        await session.flush()
        return result.rowcount > 0


class ConversationRepository:
    """对话记录数据仓库"""
    
    @staticmethod
    async def create(
        session: AsyncSession,
        room_name: str,
        user_id: str,
        role: str,
        content: str,
    ) -> Conversation:
        """创建对话记录
        
        Args:
            session: 数据库会话
            room_name: 房间名称
            user_id: 用户ID
            role: 角色（user 或 agent）
            content: 对话内容
            
        Returns:
            Conversation: 创建的对话记录
        """
        conversation = Conversation(
            room_name=room_name,
            user_id=user_id,
            role=role,
            content=content,
            created_at=datetime.now(),
        )
        session.add(conversation)
        await session.flush()
        await session.refresh(conversation)
        return conversation
    
    @staticmethod
    async def get_by_room(
        session: AsyncSession,
        room_name: str,
        limit: Optional[int] = None
    ) -> list[Conversation]:
        """根据房间名称获取对话记录
        
        Args:
            session: 数据库会话
            room_name: 房间名称
            limit: 限制返回数量
            
        Returns:
            list[Conversation]: 对话记录列表
        """
        query = select(Conversation).where(
            Conversation.room_name == room_name
        ).order_by(Conversation.created_at.asc())
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        return list(result.scalars().all())

