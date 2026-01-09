# schemas.py
from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from datetime import datetime


# --------------------
# FILES
# --------------------

class FileCreate(BaseModel):
    filename: str
    resource: Optional[str] = None
    filetype: Optional[str] = None
    public: bool = False


class FileResponse(BaseModel):
    id: int
    filename: str
    resource: Optional[str]
    filetype: Optional[str]
    public: bool
    parsed: bool
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --------------------
# PROJECTS
# --------------------

class ProjectBase(BaseModel):
    name: str


class ProjectCreate(ProjectBase):
    file1_id: int
    file2_id: int


class ProjectResponse(ProjectBase):
    id: int
    user_id: int
    file1_id: int
    file2_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --------------------
# LINKS & LINK TYPES
# --------------------

class LinkBase(BaseModel):
    project_id: int
    source_node: str
    target_node: str


class LinkCreate(LinkBase):
    link_type_id: int
    suggestion_score: float


class LinkTypeBase(BaseModel):
    group: str
    inner: str
    value: str
    public: bool = True


class LinkTypeCreate(LinkTypeBase):
    pass


class LinkTypeResponse(LinkTypeBase):
    id: int

    model_config = ConfigDict(from_attributes=True)


class LinkResponse(LinkBase):
    id: int
    user_id: int
    link_type_id: int
    suggestion_score: float
    upvote: int
    downvote: int
    created_at: datetime
    link_type: LinkTypeResponse

    model_config = ConfigDict(from_attributes=True)


# --------------------
# VOTES
# --------------------

class VoteBase(BaseModel):
    project_id: int
    link_id: int
    vote: int   # 1 = like, -1 = dislike


class VoteCreate(VoteBase):
    pass


class VoteResponse(VoteBase):
    id: int
    user_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class VoteRequest(BaseModel):
    type: str
