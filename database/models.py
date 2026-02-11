"""数据库模型"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Index
from sqlalchemy.sql import func
from database.connection import Base


class Agent(Base):
    """Agent 信息表"""
    __tablename__ = "ai_voice_agents"
    
    id = Column(Integer, primary_key=True, comment="ID")
    agent_name = Column(String(50), nullable=False, unique=True, comment="Agent名称")
    display_name = Column(String(100), nullable=False, comment="显示名称")
    created_at = Column(DateTime, nullable=False, server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now(), comment="更新时间")
    
    __table_args__ = (
        Index("idx_agent_name", "agent_name"),
    )


class Room(Base):
    """房间信息表"""
    __tablename__ = "ai_voice_rooms"
    
    id = Column(Integer, primary_key=True, comment="ID")
    room_name = Column(String(100), nullable=False, unique=True, comment="房间名称")
    room_sid = Column(String(100), comment="LiveKit房间SID")
    agent_name = Column(String(50), nullable=False, comment="Agent名称")
    user_id = Column(String(100), nullable=False, comment="用户ID")
    room_token = Column(Text, nullable=True, comment="LiveKit访问Token")
    timeout_minutes = Column(Integer, nullable=False, default=3, comment="超时时间（分钟）")
    status = Column(String(20), nullable=False, default="active", comment="状态：active, closed")
    created_at = Column(DateTime, nullable=False, server_default=func.now(), comment="创建时间")
    user_joined_at = Column(DateTime, comment="用户加入时间")
    user_left_at = Column(DateTime, comment="用户离开时间")
    chat_duration = Column(Integer, default=0, comment="聊天时长（秒）")
    closed_at = Column(DateTime, comment="房间关闭时间")
    
    __table_args__ = (
        Index("idx_room_name", "room_name"),
        Index("idx_user_id", "user_id"),
        Index("idx_agent_name", "agent_name"),
        Index("idx_status", "status"),
        Index("idx_created_at", "created_at"),
    )

