"""数据库连接配置 - Agent 端"""
import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
import logging

# ⚠️ 必须在导入前加载环境变量
from dotenv import load_dotenv
load_dotenv(".env.local")

logger = logging.getLogger(__name__)

# 创建 Base 类
Base = declarative_base()

# 数据库连接 URL
def get_database_url() -> str:
    """获取数据库连接 URL"""
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    user = os.getenv("MYSQL_USER", "ai_voice_user")
    password = os.getenv("MYSQL_PASSWORD", "")
    database = os.getenv("MYSQL_DATABASE", "ai_voice_db")
    
    logger.info(f"数据库配置: host={host}, port={port}, user={user}, database={database}")
    
    # aiomysql 连接格式: mysql+aiomysql://user:password@host:port/database
    return f"mysql+aiomysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"


# 创建异步引擎
database_url = get_database_url()
engine = create_async_engine(
    database_url,
    echo=False,  # 设置为 True 可以查看 SQL 日志
    pool_pre_ping=True,  # 连接前检查连接是否有效
    pool_recycle=3600,  # 连接回收时间（秒）
)

# 创建异步会话工厂
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    获取数据库会话
    
    Yields:
        AsyncSession: 数据库会话
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

