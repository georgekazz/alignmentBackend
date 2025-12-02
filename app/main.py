from fastapi import FastAPI, Depends, HTTPException, Form, APIRouter
from sqlalchemy.orm import Session
from app.database import get_db, engine, Base
from app.models import User,File, Project, Vote, Link, LinkType
from app.auth import hash_password, verify_password, create_access_token
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer
from app.auth import SECRET_KEY, ALGORITHM
from jose import jwt, JWTError
from fastapi import UploadFile, File as UploadFileField
from app.schemas import FileCreate, FileResponse, VoteCreate, VoteResponse, LinkCreate, LinkResponse, LinkTypeResponse, ProjectResponse, VoteRequest
import shutil
import os
from rdflib import Graph, Namespace
from rapidfuzz import fuzz
import json
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query


app = FastAPI(title="ARXIVE Auth Example")

# --- CORS middleware ---
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://192.168.6.123:8000",
]

app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"], #prosorina
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/db-test")
def test_db_connection(db: Session = Depends(get_db)):
    try:
        users_count = db.query(User).count()
        return {"status": "OK", "users_count": users_count}
    except Exception as e:
        return {"status": "ERROR", "details": str(e)}

class UserRegister(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

# Register
@app.post("/register")
def register(user: UserRegister, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_pw = hash_password(user.password)
    new_user = User(name=user.name, email=user.email, password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User registered successfully", "user_id": new_user.id}

# Login
@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    try:
        db_user = db.query(User).filter(User.email == user.email).first()
        if not db_user or not verify_password(user.password, db_user.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        token = create_access_token({"sub": db_user.email})
        return {"access_token": token, "token_type": "bearer"}
    except Exception as e:
        return {"status": "ERROR", "details": str(e)}


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

@app.get("/me")
def read_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
        "created_at": current_user.created_at
    }


from sqlalchemy.orm import Session, aliased
@app.get("/my-projects")
def get_my_projects(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    File1 = aliased(File)
    File2 = aliased(File)

    projects = (
        db.query(
            Project.id,
            Project.name,
            Project.created_at,
            File1.filename.label("file1_name"),
            File2.filename.label("file2_name")
        )
        .join(File1, Project.file1_id == File1.id, isouter=True)
        .join(File2, Project.file2_id == File2.id, isouter=True)
        .filter(Project.user_id == current_user.id)
        .all()
    )

    return [
        {
            "id": p.id,
            "name": p.name,
            "file1_name": p.file1_name,
            "file2_name": p.file2_name,
            "created_at": p.created_at
        }
        for p in projects
    ]

@app.post("/files/", response_model=FileResponse)
def create_file(file: FileCreate, db: Session = Depends(get_db)):
    db_file = File(**file.dict())
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    return db_file

@app.get("/files/", response_model=list[FileResponse])
def list_files(db: Session = Depends(get_db)):
    return db.query(File).all()

@app.get("/files/{file_id}", response_model=FileResponse)
def get_file(file_id: int, db: Session = Depends(get_db)):
    db_file = db.query(File).filter(File.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")
    return db_file

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/files/upload", response_model=FileResponse)
def upload_rdf_file(
    uploaded_file: UploadFile = UploadFileField(...),
    public: bool = Form(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    
    if not uploaded_file.filename.endswith(".rdf"):
        raise HTTPException(status_code=400, detail="Only RDF files are allowed")
    
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)

    db_file = File(
        filename=uploaded_file.filename,
        resource=file_path,
        filetype="rdf",
        public=public,
        parsed=False,
        status="new",
        owner_id=current_user.id
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    return db_file

TRIPLES_DIR = "uploads/triples"
os.makedirs(TRIPLES_DIR, exist_ok=True)

@app.post("/files/{file_id}/parse")
def rdf_to_triples(
    file_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    db_file = db.query(File).filter(File.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")

    if not db_file.filename.endswith(".rdf"):
        raise HTTPException(status_code=400, detail="File is not RDF")

    g = Graph()
    try:
        g.parse(db_file.resource)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse RDF: {str(e)}")

    triples_filename = db_file.filename.replace(".rdf", "_triples.nt")
    triples_path = os.path.join(TRIPLES_DIR, triples_filename)

    g.serialize(destination=triples_path, format="nt")

    db_file.parsed = True
    db_file.status = "parsed"
    db.commit()
    db.refresh(db_file)

    return {
        "message": "RDF file converted to triples successfully",
        "triples_file": triples_filename,
        "triples_count": len(g),
        "parsed": db_file.parsed
    }


class ProjectCreate(BaseModel):
    name: str
    file1_id: int
    file2_id: int

def nt_to_tree(nt_content: str):
    g = Graph()
    g.parse(data=nt_content, format="nt")

    nodes = {}
    root_nodes = []

    for s, p, o in g.triples((None, None, None)):
        s = str(s)
        p = str(p)
        o = str(o)

        if any(k in p for k in ["prefLabel", "altLabel", "label"]):
            label = o.strip('"').replace("@en", "")

            if label.startswith("http"):
                continue
            if s not in nodes:
                nodes[s] = {"name": label, "uri": s, "children": []}
            else:
                nodes[s]["name"] = label

    for s, p, o in g.triples((None, None, None)):
        s = str(s)
        p = str(p)
        o = str(o)
        if "broader" in p:
            parent_uri = o.strip("<>")
            child_node = nodes.get(s)
            parent_node = nodes.get(parent_uri)
            if child_node:
                if parent_node:
                    parent_node["children"].append(child_node)
                else:
                    root_nodes.append(child_node)

    for uri, node in nodes.items():
        has_parent = any(node in n.get("children", []) for n in nodes.values())
        if not has_parent and node not in root_nodes:
            root_nodes.append(node)

    return {"name": "Root", "children": root_nodes}



@app.get("/project-files/{project_id}")
def get_project_files(project_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):

    project = db.query(Project).filter(Project.id == project_id, Project.user_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    file_ids = [project.file1_id, project.file2_id]
    files_data = []

    for file_id in file_ids:
        file = db.query(File).filter(File.id == file_id).first()
        if not file:
            raise HTTPException(status_code=404, detail=f"File {file_id} not found")

        base_name = os.path.splitext(file.filename)[0]
        nt_filename = f"{base_name}_triples.nt"
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        nt_path = os.path.join(BASE_DIR, "uploads", "triples", nt_filename)
        print("Looking for NT file at:", nt_path)


        if not os.path.exists(nt_path):
            raise HTTPException(status_code=404, detail=f"{nt_filename} not found in triples folder")

        with open(nt_path, "r", encoding="utf-8") as f:
            nt_content = f.read()

        files_data.append({
            "id": file.id,
            "filename": nt_filename,
            "filetype": "nt",
            "status": file.status,
            "public": file.public,
            "created_at": file.created_at,
            "content": nt_content,
            "tree": nt_to_tree(nt_content)
        })

    return files_data
    
@app.get("/node-details/")
def node_details(project_id: int = Query(...), uri: str = Query(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    from rdflib import Graph, URIRef
    import os

    project = db.query(Project).filter(Project.id == project_id, Project.user_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    file_ids = [project.file1_id, project.file2_id]
    files_paths = []
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    for file_id in file_ids:
        file = db.query(File).filter(File.id == file_id).first()
        if file:
            base_name = os.path.splitext(file.filename)[0]
            nt_path = os.path.join(BASE_DIR, "uploads", "triples", f"{base_name}_triples.nt")
            if os.path.exists(nt_path):
                files_paths.append(nt_path)

    if not files_paths:
        raise HTTPException(status_code=404, detail="No NT files found for this project")

    details = {}
    node_uri = URIRef(uri)
    found = False

    for path in files_paths:
        g = Graph()
        g.parse(path, format="nt")
        for s, p, o in g.triples((node_uri, None, None)):
            pred = str(p).split("#")[-1] if "#" in str(p) else str(p).split("/")[-1]
            details.setdefault(pred, [])
            details[pred].append(str(o))
            found = True

    if not found:
        raise HTTPException(status_code=404, detail=f"Node {uri} not found in project files")

    for k, v in details.items():
        if len(v) == 1:
            details[k] = v[0]

    return {"uri": uri, "details": details}



from sqlalchemy import or_

@app.get("/my-files", response_model=list[FileResponse])
def get_my_files(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(File).filter(
        or_(
            File.owner_id == current_user.id,
            File.public == True
        )
    ).all()


@app.get("/projects/{project_id}", response_model=ProjectResponse)
def get_project(project_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@app.post("/projects/")
def create_project(
    project: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Έλεγχος ότι υπάρχουν τα αρχεία και είναι parsed
    file1 = db.query(File).filter(File.id == project.file1_id, File.parsed == True).first()
    file2 = db.query(File).filter(File.id == project.file2_id, File.parsed == True).first()
    if not file1 or not file2:
        raise HTTPException(status_code=400, detail="Both files must be parsed .nt files")

    new_project = Project(
        name=project.name,
        user_id=current_user.id,
        file1_id=file1.id,
        file2_id=file2.id
    )
    db.add(new_project)
    db.commit()
    db.refresh(new_project)

    return {
        "message": "Project created successfully",
        "project_id": new_project.id,
        "files": [file1.filename, file2.filename]
    }


@app.get("/link-types")
def get_link_types(group: str = None, db: Session = Depends(get_db)):
    query = db.query(LinkType)
    if group:
        query = query.filter(LinkType.group == group)
    return query.all()


SUGGESTIONS_DIR = "uploads/suggestions"
os.makedirs(SUGGESTIONS_DIR, exist_ok=True)
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

# -------------------------
#  Helper: guess RDF format
# -------------------------
def guess_rdf_format(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.read(100).lstrip()
    if first_line.startswith("<?xml"):
        return "xml"
    elif first_line.startswith("@prefix") or first_line.startswith("@base"):
        return "ttl"
    else:
        return "nt"

# -------------------------
#  Endpoint: Generate Suggestions
# -------------------------
@app.post("/projects/{project_id}/suggestions/generate")
def generate_suggestions(project_id: int, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    file1 = db.query(File).filter(File.id == project.file1_id).first()
    file2 = db.query(File).filter(File.id == project.file2_id).first()
    if not file1 or not file2:
        raise HTTPException(status_code=404, detail="Files not found")

    g1, g2 = Graph(), Graph()
    format1 = guess_rdf_format(file1.resource)
    format2 = guess_rdf_format(file2.resource)

    g1.parse(file1.resource, format=format1)
    g2.parse(file2.resource, format=format2)

    labels1 = [(str(s), str(o)) for s, o in g1.subject_objects(SKOS.prefLabel)]
    labels2 = [(str(s), str(o)) for s, o in g2.subject_objects(SKOS.prefLabel)]

    suggestions = []
    for uri1, label1 in labels1:
        for uri2, label2 in labels2:
            score = fuzz.token_sort_ratio(label1, label2) / 100.0
            if score > 0.7:  
                suggestions.append({
                    "node1": uri1,
                    "label1": label1,
                    "node2": uri2,
                    "label2": label2,
                    "similarity": round(score, 2)
                })

    suggestions_file = os.path.join(SUGGESTIONS_DIR, f"{project_id}.json")
    with open(suggestions_file, "w", encoding="utf-8") as f:
        json.dump(suggestions, f, ensure_ascii=False, indent=2)

    return {"message": "Suggestions generated", "file": suggestions_file, "count": len(suggestions)}


# -------------------------
#  Endpoint: Read Suggestions
# -------------------------
@app.get("/projects/{project_id}/suggestions/read")
def read_suggestions(project_id: int):
    suggestions_file = os.path.join(SUGGESTIONS_DIR, f"{project_id}.json")
    if not os.path.exists(suggestions_file):
        raise HTTPException(status_code=404, detail="Suggestions file not found. Generate first.")
    with open(suggestions_file, "r", encoding="utf-8") as f:
        suggestions = json.load(f)
    return {"suggestions": suggestions}

SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

@app.get("/projects/{project_id}/suggestions")
def generate_node_suggestions(
    project_id: int,
    node_uri: str = Query(..., description="URI of the selected node"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    file1 = db.query(File).filter(File.id == project.file1_id).first()
    file2 = db.query(File).filter(File.id == project.file2_id).first()
    if not file1 or not file2:
        raise HTTPException(status_code=404, detail="Files not found")

    g1, g2 = Graph(), Graph()
    g1.parse(file1.resource, format="xml") 
    g2.parse(file2.resource, format="xml")

    node_label = None
    for s, o in g1.subject_objects(SKOS.prefLabel):
        if str(s) == node_uri:
            node_label = str(o)
            break
    if not node_label:
        raise HTTPException(status_code=404, detail="Node not found in left ontology")

    suggestions = []
    for s2, label2 in g2.subject_objects(SKOS.prefLabel):
        score = fuzz.token_sort_ratio(node_label, str(label2)) / 100.0
        if score > 0:  
            suggestions.append({
                "node2": str(s2),
                "label2": str(label2),
                "similarity": round(score, 2)
            })

    suggestions.sort(key=lambda x: x["similarity"], reverse=True)

    return {"node": node_uri, "label": node_label, "suggestions": suggestions}


@app.post("/projects/{project_id}/vote", response_model=VoteResponse)
def vote_link(
    project_id: int,
    vote_data: VoteCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    
    project = db.query(Project).filter(Project.id == project_id, Project.public == True).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or not public")

    existing_vote = db.query(Vote).filter(
        Vote.user_id == current_user.id,
        Vote.project_id == project_id,
        Vote.link_id == vote_data.link_id
    ).first()

    if existing_vote:
        existing_vote.vote = vote_data.vote 
    else:
        new_vote = Vote(
            user_id=current_user.id,
            project_id=project_id,
            link_id=vote_data.link_id,
            vote=vote_data.vote
        )
        db.add(new_vote)

    db.commit()
    db.refresh(existing_vote if existing_vote else new_vote)

    return existing_vote if existing_vote else new_vote


@app.get("/links")
def get_links(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    links = db.query(Link).filter(Link.user_id == current_user.id).all()
    return links

@app.get("/projects/{project_id}/links/{link_id}/score")
def get_link_score(project_id: int, link_id: int, db: Session = Depends(get_db)):
    votes = db.query(Vote).filter(
        Vote.project_id == project_id,
        Vote.link_id == link_id
    ).all()
    score = sum(v.vote for v in votes)  # like = +1, dislike = -1
    return {"link_id": link_id, "score": score, "likes": sum(1 for v in votes if v.vote == 1), "dislikes": sum(1 for v in votes if v.vote == -1)}


@app.post("/links/", response_model=LinkResponse)
def create_link(link: LinkCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    print("Received data:", link.dict())
    db_link = Link(
        project_id=link.project_id,
        user_id=current_user.id,
        source_node=link.source_node,
        target_node=link.target_node,
        link_type_id=link.link_type_id,
        suggestion_score=link.suggestion_score,
        upvote=0,
        downvote=0
    )
    db.add(db_link)
    db.commit()
    db.refresh(db_link)

    return db_link

from sqlalchemy.orm import joinedload
@app.get("/links-vote", response_model=List[LinkResponse])
def get_links(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    links = db.query(Link).options(joinedload(Link.link_type)).all()
    return links


@app.get("/user-links/")
def get_user_links(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    links = db.query(Link).filter(Link.user_id == current_user.id).all()
    return links


@app.delete("/links/{link_id}")
def delete_link(link_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    link = db.query(Link).filter(Link.id == link_id, Link.user_id == current_user.id).first()
    if not link:
        return {"error": "Link not found or you don't have permission"}
    db.delete(link)
    db.commit()
    return {"message": "Link deleted"}



@app.post("/links/{link_id}/vote")
def vote_link(link_id: int, vote: VoteRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    link = db.query(Link).filter(Link.id == link_id).first()
    if not link:
        raise HTTPException(status_code=404, detail="Link not found")

    if vote.type == "upvote":
        link.upvote += 1
    elif vote.type == "downvote":
        link.downvote += 1
    else:
        raise HTTPException(status_code=400, detail="Invalid vote type")

    db.commit()
    db.refresh(link)

    return {"upvote": link.upvote, "downvote": link.downvote}


@app.get("/link-types/", response_model=List[LinkTypeResponse])
def get_link_types(db: Session = Depends(get_db)):
    return db.query(LinkType).filter(LinkType.public == True).all()

@app.get("/projects/{project_id}/links", response_model=list[LinkResponse])
def list_links(project_id: int, db: Session = Depends(get_db)):
    return db.query(Link).filter(Link.project_id == project_id).all()

from fastapi.responses import JSONResponse
@app.get("/projects/{project_id}/export-links")
def export_project_links(project_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    
    links = db.query(Link).filter(Link.project_id == project_id).all()
    if not links:
        raise HTTPException(status_code=404, detail="No links found for this project")

    links_data = [
        {
            "id": link.id,
            "project_id": link.project_id,
            "user_id": link.user_id,
            "source_node": link.source_node,
            "target_node": link.target_node,
            "link_type_id": link.link_type_id,
            "suggestion_score": link.suggestion_score,
            "upvote": link.upvote,
            "downvote": link.downvote,
            "created_at": link.created_at.isoformat()
        }
        for link in links
    ]

    # Επιστρέφουμε JSON
    return JSONResponse(content=links_data)

from app.models import Link, LinkType
from rdflib import Graph, URIRef
from fastapi.responses import StreamingResponse
import io
@app.get("/projects/{project_id}/export")
def export_project(project_id: int, db: Session = Depends(get_db)):
    
    links = db.query(Link).filter(Link.project_id == project_id).all()
    if not links:
        raise HTTPException(status_code=404, detail="No links found for this project")

    link_types = db.query(LinkType).all()
    link_type_map = {lt.id: lt.inner for lt in link_types}

    g = Graph()

    for link in links:
        predicate_uri = link_type_map.get(link.link_type_id)
        if not predicate_uri:
           
            continue

        g.add((
            URIRef(link.source_node),
            URIRef(predicate_uri),
            URIRef(link.target_node)
        ))

    data = g.serialize(format="turtle")
    stream = io.BytesIO(data.encode("utf-8"))

    return StreamingResponse(
        stream,
        media_type="text/turtle",
        headers={"Content-Disposition": f"attachment; filename=project_{project_id}_ontology.ttl"}
    )


@app.get("/files/{file_id}/skos")
def get_skos_labels(file_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    file = db.query(File).filter(File.id == file_id, File.owner_id == current_user.id).first()
    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    g = Graph()
    format = "xml"
    g.parse(file.resource, format=format)

    labels = [{"subject": str(s), "label": str(o)} for s, o in g.subject_objects(SKOS.prefLabel)]
    return {"labels": labels}

# SKOS Viewer 
@app.get("/files/{file_id}/skos-tree")
def get_skos_tree(
    file_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    from rdflib import Graph, Namespace
    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file = db.query(File).filter(File.id == file_id).first()
    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    nt_path = os.path.join(BASE_DIR, "uploads", "triples", f"{os.path.splitext(file.filename)[0]}_triples.nt")
    if not os.path.exists(nt_path):
        raise HTTPException(status_code=404, detail="NT triples file not found")

    g = Graph()
    g.parse(nt_path, format="nt")
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

    concepts = {}
    PREFERRED_LANG = "el"

    for s in g.subjects(predicate=None, object=SKOS.Concept):
        label = None
        for _, _, l in g.triples((s, SKOS.prefLabel, None)):
            if hasattr(l, "language") and l.language == PREFERRED_LANG:
                label = str(l)
                break
            elif not label:
                label = str(l)
        concepts[str(s)] = {"uri": str(s), "label": label or str(s), "broader": [], "narrower": []}

    for s, _, o in g.triples((None, SKOS.broader, None)):
        if str(s) in concepts and str(o) in concepts:
            concepts[str(s)]["broader"].append(str(o))
            concepts[str(o)]["narrower"].append(str(s))

    roots_uris = set()
    for _, _, o in g.triples((None, SKOS.hasTopConcept, None)):
        roots_uris.add(str(o))
    for s, _, _ in g.triples((None, SKOS.topConceptOf, None)):
        roots_uris.add(str(s))
    if not roots_uris:
        roots_uris = {c["uri"] for c in concepts.values() if not c["broader"]}

    def build_node(concept, visited=None):
        if visited is None:
            visited = set()
        if concept["uri"] in visited:
            return {
                "label": concept["label"] + " (κυκλική αναφορά)",
                "uri": concept["uri"],
                "details": concept,
                "children": []
            }

        visited.add(concept["uri"])

        children = [build_node(concepts[c], visited.copy()) for c in concept["narrower"] if c in concepts]

        details = {}
        SKOS_PROPS = [
            SKOS.prefLabel, SKOS.altLabel, SKOS.definition, SKOS.scopeNote,
            SKOS.example, SKOS.notation, SKOS.related
        ]
        for p in SKOS_PROPS:
            values = [str(o) for _, _, o in g.triples((URIRef(concept["uri"]), p, None))]
            if values:
                details[str(p).split("#")[-1]] = values if len(values) > 1 else values[0]

        return {
            "label": concept["label"],
            "uri": concept["uri"],
            "details": details,
            "children": children
        }

    tree = [build_node(concepts[uri]) for uri in roots_uris if uri in concepts]
    return tree





# SILK------------------------------------------------------------------------

# import os
# import json
# import re
# from rapidfuzz import fuzz
# from nltk.stem import SnowballStemmer
# from typing import List
# from rdflib.namespace import RDFS
# from rdflib import URIRef
# from sentence_transformers import SentenceTransformer, util
# import torch

# # --- Config ---
# SUGGESTIONS_DIR = "uploads/suggestions"
# SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

# # NLP setup
# stemmer = SnowballStemmer("english")
# STOPWORDS = set("""
# a able about across after all almost also am among an and any are as at be because been but by can cannot could dear did do does either else ever every for from get got had has have he her hers him his how however i if in into is it its just least let like likely may me might most must my neither no nor not of off often on only or other our own rather said say says she should since so some than that the their them then there these they this tis to too twas us wants was we were what when where which while who whom why will with would yet you your
# """.split())

# model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")


# def clean_label(label: str) -> str:
#     label = label.strip()
#     label = re.sub(r"\s+", " ", label)
#     return label


# def preprocess_label(label: str) -> str:
#     label = label.lower()
#     label = re.sub(r"[^a-zA-Zα-ωΑ-Ω0-9\s\-]", " ", label)
#     tokens = [stemmer.stem(t) for t in label.split() if t not in STOPWORDS]
#     return " ".join(tokens)


# @app.get("/projects/{project_id}/suggestions_full")
# def generate_node_suggestions_full_embeddings(
#     project_id: int,
#     node_uri: str = Query(..., description="URI of the selected node"),
#     db: Session = Depends(get_db),
#     current_user=Depends(get_current_user)
# ):
#     project = (
#         db.query(Project)
#         .filter(Project.id == project_id, Project.user_id == current_user.id)
#         .first()
#     )
#     if not project:
#         raise HTTPException(status_code=404, detail="Project not found")

#     file1 = db.query(File).filter(File.id == project.file1_id).first()
#     file2 = db.query(File).filter(File.id == project.file2_id).first()

#     if not file1 or not file2:
#         raise HTTPException(status_code=404, detail="Files not found")

#     format1 = guess_rdf_format(file1.resource)
#     format2 = guess_rdf_format(file2.resource)

#     g1, g2 = Graph(), Graph()
#     g1.parse(file1.resource, format=format1)
#     g2.parse(file2.resource, format=format2)

#     node_props = [SKOS.prefLabel, SKOS.altLabel, RDFS.label]
#     node_uri_ref = URIRef(node_uri)

#     node_labels_original = [
#         clean_label(str(l))
#         for prop in node_props
#         for l in g1.objects(node_uri_ref, prop)
#     ]

#     node_labels_processed = [
#         preprocess_label(str(l))
#         for prop in node_props
#         for l in g1.objects(node_uri_ref, prop)
#     ]

#     if not node_labels_processed:
#         raise HTTPException(status_code=404, detail="Node not found in left ontology")

#     node_embs = model.encode(
#         node_labels_processed,
#         convert_to_tensor=True,
#         normalize_embeddings=True
#     )

#     suggestions = []
#     seen_labels = set()
#     target_props = [SKOS.prefLabel, SKOS.altLabel]

#     for s2 in g2.subjects():

#         target_original_labels = [
#             clean_label(str(l))
#             for prop in target_props
#             for l in g2.objects(s2, prop)
#         ]

#         target_processed_labels = [
#             preprocess_label(str(l))
#             for prop in target_props
#             for l in g2.objects(s2, prop)
#         ]

#         if not target_processed_labels:
#             continue

#         target_embs = model.encode(
#             target_processed_labels,
#             convert_to_tensor=True,
#             normalize_embeddings=True
#         )

#         best_score = 0.0
#         best_label_original = None

#         for nl_emb in node_embs:
#             sim_row = util.cos_sim(nl_emb, target_embs)
#             max_val, max_idx = torch.max(sim_row, dim=1)

#             score = max_val.item()
#             idx = max_idx.item()

#             if score > best_score:
#                 best_score = score
#                 best_label_original = target_original_labels[idx]

#         if best_label_original and best_label_original not in seen_labels:
#             suggestions.append({
#                 "node2": str(s2),
#                 "label2": best_label_original,
#                 "similarity": round(best_score, 2)
#             })
#             seen_labels.add(best_label_original)

#     suggestions.sort(key=lambda x: x["similarity"], reverse=True)

#     os.makedirs(SUGGESTIONS_DIR, exist_ok=True)
#     output_file = os.path.join(SUGGESTIONS_DIR, f"{project_id}.json")

#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(suggestions, f, ensure_ascii=False, indent=2)

#     return {
#         "node": node_uri,
#         "labels": node_labels_original,
#         "suggestions": suggestions
#     }



# SILK------------------------------------------------------------------------

import os
import json
import re
from rapidfuzz import fuzz
from nltk.stem import SnowballStemmer
from typing import List
from rdflib.namespace import RDFS
from rdflib import URIRef

SUGGESTIONS_DIR = "uploads/suggestions"
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

# NLP setup
stemmer = SnowballStemmer("english")
STOPWORDS = set("""
a able about across after all almost also am among an and any are as at be because been but by can cannot could dear did do does either else ever every for from get got had has have he her hers him his how however i if in into is it its just least let like likely may me might most must my neither no nor not of off often on only or other our own rather said say says she should since so some than that the their them then there these they this tis to too twas us wants was we were what when where which while who whom why will with would yet you your
""".split())

def preprocess_label(label: str) -> str:
    label = label.lower()
    label = re.sub(r"[^a-zα-ω0-9\s]", " ", label, flags=re.IGNORECASE)
    tokens = [stemmer.stem(t) for t in label.split() if t not in STOPWORDS]
    return " ".join(tokens)

def dice_similarity(a: str, b: str) -> float:
    def bigrams(s):
        s = s.lower()
        return {s[i:i+2] for i in range(len(s)-1)}
    
    a_bigrams = bigrams(a)
    b_bigrams = bigrams(b)
    if not a_bigrams or not b_bigrams:
        return 0.0
    overlap = len(a_bigrams & b_bigrams)
    return 2 * overlap / (len(a_bigrams) + len(b_bigrams))


@app.get("/projects/{project_id}/suggestions_full")
def generate_node_suggestions_full(
    project_id: int,
    node_uri: str = Query(..., description="URI of the selected node"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
   
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    file1 = db.query(File).filter(File.id == project.file1_id).first()
    file2 = db.query(File).filter(File.id == project.file2_id).first()
    if not file1 or not file2:
        raise HTTPException(status_code=404, detail="Files not found")

    format1 = guess_rdf_format(file1.resource)
    format2 = guess_rdf_format(file2.resource)
    g1, g2 = Graph(), Graph()
    g1.parse(file1.resource, format=format1)
    g2.parse(file2.resource, format=format2)

    node_props = [SKOS.prefLabel, SKOS.altLabel, RDFS.label]
    node_uri_ref = URIRef(node_uri)
    node_label = None
    print(f"Node URI: {node_uri_ref}")
    for p, o in g1.predicate_objects(subject=node_uri_ref):
        print("Predicate:", p, "| Object:", o)

    for prop in node_props:
        val = g1.value(subject=node_uri_ref, predicate=prop)
        if val:
            node_label = preprocess_label(str(val))
            break

    if not node_label:
        raise HTTPException(status_code=404, detail="Node not found in left ontology")

    suggestions = []
    target_props = [SKOS.prefLabel, SKOS.altLabel]
    seen_labels = set()

    for s2 in g2.subjects():
        best_score = 0
        best_label = None
        for prop in target_props:
            label2 = g2.value(subject=s2, predicate=prop)
            if label2:
                processed_label2 = preprocess_label(str(label2))
                score = max(dice_similarity(node_label, processed_label2),
                            fuzz.token_sort_ratio(node_label, processed_label2)/100)
                if score > best_score:
                    best_score = score
                    best_label = str(label2)
        
        if best_score > 0 and best_label not in seen_labels:
            suggestions.append({
                "node2": str(s2),
                "label2": best_label,
                "similarity": round(best_score, 2)
            })
            seen_labels.add(best_label)

    suggestions.sort(key=lambda x: x["similarity"], reverse=True)

    os.makedirs(SUGGESTIONS_DIR, exist_ok=True)
    output_file = os.path.join(SUGGESTIONS_DIR, f"{project_id}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(suggestions, f, ensure_ascii=False, indent=2)

    return {"node": node_uri, "label": node_label, "suggestions": suggestions}