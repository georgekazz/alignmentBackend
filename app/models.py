from app.database import Base
from datetime import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, Float, DateTime, func, Boolean
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    email_verified_at = Column(DateTime, nullable=True)
    password = Column(String(255), nullable=False)
    remember_token = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    resource = Column(String(500))
    filetype = Column(String(50))
    public = Column(Boolean, default=False)
    parsed = Column(Boolean, default=False)
    status = Column(String(50), default="new")
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    file1_id = Column(Integer, ForeignKey("files.id"), nullable=False)
    file2_id = Column(Integer, ForeignKey("files.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Σχέσεις (προαιρετικές)
    user = relationship("User", backref="projects")
    file1 = relationship("File", foreign_keys=[file1_id])
    file2 = relationship("File", foreign_keys=[file2_id])


class Vote(Base):
    __tablename__ = "votes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    link_id = Column(Integer, nullable=False)
    vote = Column(Integer, nullable=False)  # 1 = like, -1 = dislike
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Link(Base):
    __tablename__ = "links"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, index=True)
    user_id = Column(Integer, index=True)
    source_node = Column(String(255))
    target_node = Column(String(255))
    link_type_id = Column(Integer, ForeignKey("link_types.id"))
    suggestion_score = Column(Integer)
    upvote = Column(Integer, default=0)
    downvote = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    link_type = relationship("LinkType", back_populates="links")

class LinkType(Base):
    __tablename__ = "link_types"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, default=0)
    group = Column(String(50))             # π.χ. "SKOS"
    inner = Column(String(100))            # π.χ. "Exact Match"
    value = Column(String(255))            # π.χ. "http://www.w3.org/2004/02/skos/core#exactMatch"
    public = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    links = relationship("Link", back_populates="link_type")