# schemas.py
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from typing import List

class FileCreate(BaseModel):
    filename: str
    resource: Optional[str] = None
    filetype: Optional[str] = None
    public: Optional[bool] = False

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

    class Config:
        orm_mode = True

class ProjectBase(BaseModel):
    name: str

class ProjectCreate(ProjectBase):
    file1_id: int
    file2_id: int

class ProjectResponse(ProjectBase):
    id: int
    name: str
    user_id: int
    file1_id: int
    file2_id: int
    created_at: datetime

    class Config:
        orm_mode = True

class VoteCreate(BaseModel):
    link_id: int
    vote: int 

class VoteResponse(BaseModel):
    link_id: int
    vote: int
    user_id: int
    project_id: int

    class Config:
        orm_mode = True


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

    class Config:
        orm_mode = True


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

class LinkTypeResponse(LinkTypeBase):
    id: int

    class Config:
        orm_mode = True

class LinkResponse(LinkBase):
    id: int
    user_id: int
    link_type_id: int
    suggestion_score: float
    upvote: int
    downvote: int
    created_at: datetime
    link_type: LinkTypeResponse
    class Config:
        orm_mode = True


class LinkTypeCreate(LinkTypeBase):
    pass



class VoteRequest(BaseModel):
    type: str