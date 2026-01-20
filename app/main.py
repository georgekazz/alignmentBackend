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
    "http://alignment.okfn.gr",
]

app.add_middleware(
    CORSMiddleware,
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
    # Get file
    db_file = db.query(File).filter(File.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")

    # Check if file is RDF-compatible
    rdf_extensions = [".rdf", ".ttl", ".owl", ".n3", ".nt", ".xml", ".jsonld"]
    file_ext = os.path.splitext(db_file.filename)[1].lower()
    
    if file_ext not in rdf_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File format not supported. Supported: {', '.join(rdf_extensions)}"
        )

    os.makedirs(TRIPLES_DIR, exist_ok=True)

    g = Graph()
    try:
        format_map = {
            ".rdf": "xml",
            ".owl": "xml",
            ".ttl": "turtle",
            ".n3": "n3",
            ".nt": "nt",
            ".xml": "xml",
            ".jsonld": "json-ld"
        }
        rdf_format = format_map.get(file_ext, "xml")
        
        if not os.path.exists(db_file.resource):
            raise HTTPException(status_code=404, detail="File resource not found on disk")
        
        g.parse(db_file.resource, format=rdf_format)
        
    except Exception as e:
        db_file.status = "error"
        db.commit()
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to parse RDF file: {str(e)}"
        )

    if len(g) == 0:
        raise HTTPException(
            status_code=400, 
            detail="Parsed graph is empty. File may be invalid or corrupted."
        )

    base_name = os.path.splitext(db_file.filename)[0]
    base_name = "".join(c for c in base_name if c.isalnum() or c in ('-', '_'))
    triples_filename = f"{base_name}_triples.nt"
    triples_path = os.path.join(TRIPLES_DIR, triples_filename)

    try:
        g.serialize(destination=triples_path, format="nt", encoding="utf-8")
    except Exception as e:
        db_file.status = "error"
        db.commit()
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to serialize triples: {str(e)}"
        )

    db_file.parsed = True
    db_file.status = "parsed"
    db.commit()
    db.refresh(db_file)

    return {
        "message": "RDF file converted to triples successfully",
        "triples_file": triples_filename,
        "triples_path": triples_path,
        "triples_count": len(g),
        "parsed": db_file.parsed,
        "source_format": rdf_format
    }


class ProjectCreate(BaseModel):
    name: str
    file1_id: int
    file2_id: int

from rdflib import Graph, Namespace, URIRef, RDFS
from typing import Dict, List, Set
import re

SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
OWL = Namespace("http://www.w3.org/2002/07/owl#")

def nt_to_tree(nt_content: str) -> Dict:
   
    g = Graph()
    try:
        g.parse(data=nt_content, format="nt")
    except Exception as e:
        raise ValueError(f"Failed to parse N-Triples: {str(e)}")
    
    if len(g) == 0:
        return {"name": "Root", "uri": "root", "children": []}
    
    LABEL_PREDICATES = {
        str(SKOS.prefLabel),
        str(SKOS.altLabel),
        str(RDFS.label)
    }
    
    BROADER_PREDICATES = {
        str(SKOS.broader),
    }
    
    NARROWER_PREDICATES = {
        str(SKOS.narrower),
    }
    
    SUBCLASS_PREDICATES = {
        str(RDFS.subClassOf),
        str(OWL.subClassOf)
    }
    
    TOP_CONCEPT_PREDICATES = {
        str(SKOS.hasTopConcept),
        str(SKOS.topConceptOf)
    }
    
    nodes: Dict[str, Dict] = {}
    child_to_parents: Dict[str, Set[str]] = {}
    parent_to_children: Dict[str, List[str]] = {}
    explicit_top_concepts: Set[str] = set()
    concept_scheme_uri: str = None
    concept_scheme_label: str = None
    
    for s, p, o in g.triples((None, None, None)):
        s_str = str(s)
        p_str = str(p)
        o_str = str(o)
        
        if p_str not in LABEL_PREDICATES:
            continue
        
        if not s_str.startswith("http"):
            continue
        
        label = o_str
        
        if label.startswith('"') and '"' in label[1:]:
            label = label[1:label.rindex('"')]
        
        label = re.sub(r'@[a-z]{2}(-[A-Z]{2})?$', '', label).strip()
        
        if label.startswith("http"):
            continue
        
        if not label:
            continue
        
        if s_str not in nodes:
            nodes[s_str] = {
                "name": label,
                "uri": s_str,
                "children": []
            }
        else:
            if "name" not in nodes[s_str] or not nodes[s_str]["name"]:
                nodes[s_str]["name"] = label
    
    for s in g.subjects(predicate=None, object=SKOS.ConceptScheme):
        concept_scheme_uri = str(s)
        for pred in [SKOS.prefLabel, RDFS.label]:
            label = g.value(subject=s, predicate=pred)
            if label:
                concept_scheme_label = str(label)
                concept_scheme_label = re.sub(r'@[a-z]{2}(-[A-Z]{2})?$', '', concept_scheme_label).strip('"')
                break
        break  
    
    for s, p, o in g.triples((None, None, None)):
        p_str = str(p)
        
        if p_str == str(SKOS.hasTopConcept):
            explicit_top_concepts.add(str(o))
        elif p_str == str(SKOS.topConceptOf):
            explicit_top_concepts.add(str(s))
    
    for s, p, o in g.triples((None, None, None)):
        s_str = str(s)
        p_str = str(p)
        o_str = str(o)
        
        if not s_str.startswith("http") or not o_str.startswith("http"):
            continue
        
        child_uri = None
        parent_uri = None
        
        if p_str in BROADER_PREDICATES:
            child_uri = s_str
            parent_uri = o_str
        
        elif p_str in NARROWER_PREDICATES:
            parent_uri = s_str
            child_uri = o_str
        
        elif p_str in SUBCLASS_PREDICATES:
            child_uri = s_str
            parent_uri = o_str
        
        if child_uri and parent_uri:
            if child_uri not in child_to_parents:
                child_to_parents[child_uri] = set()
            child_to_parents[child_uri].add(parent_uri)
            
            if parent_uri not in parent_to_children:
                parent_to_children[parent_uri] = []
            if child_uri not in parent_to_children[parent_uri]:
                parent_to_children[parent_uri].append(child_uri)
    
    for parent_uri in parent_to_children.keys():
        if parent_uri not in nodes:
            label = parent_uri.split("#")[-1].split("/")[-1]
            label = re.sub(r'[_-]', ' ', label)
            
            nodes[parent_uri] = {
                "name": label,
                "uri": parent_uri,
                "children": []
            }
    
    for parent_uri, child_uris in parent_to_children.items():
        if parent_uri not in nodes:
            continue
        
        parent_node = nodes[parent_uri]
        
        for child_uri in child_uris:
            if child_uri not in nodes:
                continue
            
            child_node = nodes[child_uri]
            
            child_id = id(child_node)
            parent_children_ids = {id(c) for c in parent_node["children"]}
            
            if child_id not in parent_children_ids:
                parent_node["children"].append(child_node)
    
    root_nodes = []
    root_node_ids = set()
    
    if explicit_top_concepts:
        for uri in explicit_top_concepts:
            if uri in nodes:
                node = nodes[uri]
                node_id = id(node)
                if node_id not in root_node_ids:
                    root_nodes.append(node)
                    root_node_ids.add(node_id)
    
    if not root_nodes:
        for uri, node in nodes.items():
            has_parent = uri in child_to_parents and len(child_to_parents[uri]) > 0
            
            if not has_parent:
                node_id = id(node)
                if node_id not in root_node_ids:
                    root_nodes.append(node)
                    root_node_ids.add(node_id)
    
    if not root_nodes:
        nodes_by_children = sorted(
            nodes.items(),
            key=lambda x: len(parent_to_children.get(x[0], [])),
            reverse=True
        )
        
        for uri, node in nodes_by_children[:min(20, len(nodes_by_children))]:
            node_id = id(node)
            if node_id not in root_node_ids:
                root_nodes.append(node)
                root_node_ids.add(node_id)
    
    if not root_nodes and nodes:
        root_nodes = list(nodes.values())
    
    root_name = "Root"
    root_uri = "root"
    
    if concept_scheme_label:
        root_name = concept_scheme_label
        root_uri = concept_scheme_uri or "root"
    elif len(root_nodes) == 1:
        root_name = root_nodes[0]["name"]
        root_uri = root_nodes[0]["uri"]
        root_nodes = root_nodes[0]["children"]
    else:
        common_words = set()
        for node in root_nodes[:5]:
            words = set(node["name"].lower().split())
            if not common_words:
                common_words = words
            else:
                common_words &= words
        
        if common_words:
            root_name = " ".join(sorted(common_words)).title() + " Ontology"
    
    return {
        "name": root_name,
        "uri": root_uri,
        "children": root_nodes
    }


@app.delete("/files/{file_id}")
def delete_file(
    file_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):

    file = db.query(File).filter(
        File.id == file_id,
        File.owner_id == current_user.id
    ).first()

    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    is_used = db.query(Project).filter(
        (Project.file1_id == file_id) | (Project.file2_id == file_id)
    ).first()

    if is_used:
        raise HTTPException(status_code=400, detail="File is used in a project and cannot be deleted")

    if os.path.exists(file.resource):
        os.remove(file.resource)

    db.delete(file)
    db.commit()

    return {"message": "File deleted successfully"}


@app.get("/project-files/{project_id}")
def get_project_files(
    project_id: int, 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    
    project = db.query(Project).filter(
        Project.id == project_id, 
        Project.user_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if not project.file1_id or not project.file2_id:
        raise HTTPException(
            status_code=400, 
            detail="Project must have two files assigned"
        )
    
    file_ids = [project.file1_id, project.file2_id]
    
    files = db.query(File).filter(File.id.in_(file_ids)).all()
    
    if len(files) != 2:
        missing_ids = set(file_ids) - {f.id for f in files}
        raise HTTPException(
            status_code=404, 
            detail=f"Files not found: {missing_ids}"
        )
    
    files_dict = {f.id: f for f in files}
    ordered_files = [files_dict[fid] for fid in file_ids]
    
    files_data = []
    
    for file in ordered_files:
        if not file.parsed or file.status != "parsed":
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' has not been parsed yet. Parse it first."
            )
        
        base_name = os.path.splitext(file.filename)[0]
        base_name = "".join(c for c in base_name if c.isalnum() or c in ('-', '_'))
        nt_filename = f"{base_name}_triples.nt"
        
        nt_path = os.path.join(TRIPLES_DIR, nt_filename)
        
        if not os.path.exists(nt_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Triples file not found: {nt_filename}. File may need to be re-parsed."
            )
        
        try:
            with open(nt_path, "r", encoding="utf-8") as f:
                nt_content = f.read()
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read {nt_filename}: encoding error"
            )
        except IOError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read {nt_filename}: {str(e)}"
            )
        
        try:
            tree = nt_to_tree(nt_content)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to build tree for {nt_filename}: {str(e)}"
            )
        
        files_data.append({
            "id": file.id,
            "filename": nt_filename,
            "original_filename": file.filename,
            "filetype": "nt",
            "status": file.status,
            "public": file.public,
            "created_at": file.created_at,
            "content": nt_content,
            "tree": tree,
            "triples_count": len(nt_content.split('\n')) if nt_content else 0
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
            
            if not os.path.exists(nt_path):
                nt_path = os.path.join("/app/uploads/triples", f"{base_name}_triples.nt")
            
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
from datetime import datetime
@app.get("/projects/{project_id}/export-links")
def export_project_links_json(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    links = db.query(Link).filter(Link.project_id == project_id).all()
    
    if not links:
        raise HTTPException(status_code=404, detail="No links found for this project")
    
    links_data = []
    for link in links:
        
        links_data.append({
            "id": link.id,
            "source_node": link.source_node,
            "target_node": link.target_node,
            "link_type_id": link.link_type_id,
            "suggestion_score": link.suggestion_score,
            "upvote": link.upvote,
            "downvote": link.downvote,
            "created_at": link.created_at.isoformat() if link.created_at else None
        })
    
    headers = {
        "Content-Disposition": f"attachment; filename=project_{project_id}_links.json"
    }
    
    return JSONResponse(
        content={
            "project_id": project_id,
            "project_name": project.name,
            "user_id": current_user.id,
            "total_links": len(links),
            "export_date": datetime.now().isoformat(),
            "links": links_data
        },
        headers=headers
    )

from app.models import Link, LinkType
from rdflib import Graph, URIRef
from fastapi.responses import StreamingResponse
import io
@app.get("/projects/{project_id}/export")
def export_project(
    project_id: int,
    format: str = "turtle",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
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
    
    valid_formats = {
        "turtle": ("text/turtle", "ttl"),
        "xml": ("application/rdf+xml", "rdf"),
        "nt": ("application/n-triples", "nt"),
        "n3": ("text/n3", "n3")
    }
    
    if format not in valid_formats:
        format = "turtle"  
    
    media_type, extension = valid_formats[format]
    
    data = g.serialize(format=format)
    stream = io.BytesIO(data.encode("utf-8"))
    
    return StreamingResponse(
        stream,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename=project_{project_id}_ontology.{extension}"
        }
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

    base_name = os.path.splitext(file.filename)[0]

    nt_path = os.path.join(BASE_DIR, "uploads", "triples", f"{base_name}_triples.nt")

    if not os.path.exists(nt_path):
        nt_path = os.path.join("/app/uploads/triples", f"{base_name}_triples.nt")

    if not os.path.exists(nt_path):
        raise HTTPException(status_code=404, detail=f"NT triples file {nt_path} not found")

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

import os
import json
import re
from rapidfuzz import fuzz
from nltk.stem import SnowballStemmer
from typing import List
from rdflib.namespace import RDFS
from rdflib import URIRef
from typing import List, Dict, Set, Tuple

SUGGESTIONS_DIR = "uploads/suggestions"
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
os.makedirs(SUGGESTIONS_DIR, exist_ok=True)

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

# NLP setup
stemmer = SnowballStemmer("english")
STOPWORDS = set("""
a able about across after all almost also am among an and any are as at be because been but by can cannot could dear did do does either else ever every for from get got had has have he her hers him his how however i if in into is it its just least let like likely may me might most must my neither no nor not of off often on only or other our own rather said say says she should since so some than that the their them then there these they this tis to too twas us wants was we were what when where which while who whom why will with would yet you your
""".split())

def preprocess_label(label: str) -> str:
    """Normalize and stem a label for comparison."""
    label = label.lower()
    label = re.sub(r"[^a-zα-ω0-9\s]", " ", label, flags=re.IGNORECASE)
    tokens = [stemmer.stem(t) for t in label.split() if t not in STOPWORDS]
    return " ".join(tokens)

def dice_similarity(a: str, b: str) -> float:
    """Calculate Dice coefficient based on character bigrams."""
    def bigrams(s):
        s = s.lower()
        return {s[i:i+2] for i in range(len(s)-1)}
    
    a_bigrams = bigrams(a)
    b_bigrams = bigrams(b)
    if not a_bigrams or not b_bigrams:
        return 0.0
    overlap = len(a_bigrams & b_bigrams)
    return 2 * overlap / (len(a_bigrams) + len(b_bigrams))

def get_all_labels(graph, uri_ref) -> List[str]:
    """Extract all labels (prefLabel, altLabel, label) for a concept."""
    labels = []
    label_props = [SKOS.prefLabel, SKOS.altLabel, RDFS.label]
    
    for prop in label_props:
        for label in graph.objects(subject=uri_ref, predicate=prop):
            labels.append(str(label))
    
    return labels

def get_definition(graph, uri_ref) -> str:
    """Extract definition/description for a concept."""
    definition = graph.value(subject=uri_ref, predicate=SKOS.definition)
    return str(definition) if definition else ""

def get_parents(graph, uri_ref) -> Set[URIRef]:
    """Get all parent concepts (broader relations)."""
    parents = set()
    for parent in graph.objects(subject=uri_ref, predicate=SKOS.broader):
        parents.add(parent)
    return parents

def get_children(graph, uri_ref) -> Set[URIRef]:
    """Get all child concepts (narrower relations)."""
    children = set()
    for child in graph.objects(subject=uri_ref, predicate=SKOS.narrower):
        children.add(child)
    # Also check inverse broader relations
    for child in graph.subjects(predicate=SKOS.broader, object=uri_ref):
        children.add(child)
    return children

def get_siblings(graph, uri_ref) -> Set[URIRef]:
    """Get sibling concepts (sharing same parent)."""
    siblings = set()
    parents = get_parents(graph, uri_ref)
    
    for parent in parents:
        # Get all children of this parent
        for sibling in get_children(graph, parent):
            if sibling != uri_ref:
                siblings.add(sibling)
    
    return siblings

def calculate_label_similarity(labels1: List[str], labels2: List[str]) -> Tuple[float, bool]:

    max_score = 0.0
    is_exact = False
    
    for l1 in labels1:
        processed_l1 = preprocess_label(l1)
        l1_clean = l1.lower().strip()
        
        for l2 in labels2:
            processed_l2 = preprocess_label(l2)
            l2_clean = l2.lower().strip()
            
            # Check for exact match (before preprocessing)
            if l1_clean == l2_clean:
                return 1.0, True
            
            # Check for exact match after preprocessing
            if processed_l1 == processed_l2 and processed_l1:
                is_exact = True
                max_score = 1.0
                continue
            
            dice_score = dice_similarity(processed_l1, processed_l2)
            fuzz_score = fuzz.token_sort_ratio(processed_l1, processed_l2) / 100
            score = max(dice_score, fuzz_score)
            
            if score > max_score:
                max_score = score
    
    return max_score, is_exact

def calculate_definition_similarity(def1: str, def2: str) -> float:
    """Calculate similarity between definitions."""
    if not def1 or not def2:
        return 0.0
    
    processed_def1 = preprocess_label(def1)
    processed_def2 = preprocess_label(def2)
    
    if not processed_def1 or not processed_def2:
        return 0.0
    
    # Use token sort ratio for longer text
    return fuzz.token_set_ratio(processed_def1, processed_def2) / 100

def calculate_structural_similarity(
    graph1, uri1: URIRef, 
    graph2, uri2: URIRef,
    similarity_threshold: float = 0.65
) -> Dict[str, float]:
    """Calculate structural similarity based on hierarchy."""
    
    parents1 = get_parents(graph1, uri1)
    parents2 = get_parents(graph2, uri2)
    children1 = get_children(graph1, uri1)
    children2 = get_children(graph2, uri2)
    siblings1 = get_siblings(graph1, uri1)
    siblings2 = get_siblings(graph2, uri2)
    
    # Parent similarity
    parent_score = 0.0
    if parents1 and parents2:
        parent_matches = 0
        for p1 in parents1:
            p1_labels = get_all_labels(graph1, p1)
            for p2 in parents2:
                p2_labels = get_all_labels(graph2, p2)
                similarity, _ = calculate_label_similarity(p1_labels, p2_labels)
                if similarity > similarity_threshold:
                    parent_matches += 1
                    break
        parent_score = parent_matches / max(len(parents1), len(parents2))
    
    # Child similarity
    child_score = 0.0
    if children1 and children2:
        child_matches = 0
        for c1 in children1:
            c1_labels = get_all_labels(graph1, c1)
            for c2 in children2:
                c2_labels = get_all_labels(graph2, c2)
                similarity, _ = calculate_label_similarity(c1_labels, c2_labels)
                if similarity > similarity_threshold:
                    child_matches += 1
                    break
        child_score = child_matches / max(len(children1), len(children2))
    
    # Sibling similarity
    sibling_score = 0.0
    if siblings1 and siblings2:
        sibling_matches = 0
        for s1 in siblings1:
            s1_labels = get_all_labels(graph1, s1)
            for s2 in siblings2:
                s2_labels = get_all_labels(graph2, s2)
                similarity, _ = calculate_label_similarity(s1_labels, s2_labels)
                if similarity > similarity_threshold:
                    sibling_matches += 1
                    break
        sibling_score = sibling_matches / max(len(siblings1), len(siblings2))
    
    return {
        "parent": parent_score,
        "child": child_score,
        "sibling": sibling_score
    }

def calculate_combined_score(
    label_score: float,
    is_exact_match: bool,
    definition_score: float,
    structural_scores: Dict[str, float]
) -> float:
    
    if is_exact_match:
        # Base score is very high for exact matches
        base_score = 0.95
        
        definition_bonus = definition_score * 0.02
        structural_bonus = (
            structural_scores["parent"] * 0.01 +
            structural_scores["child"] * 0.01 +
            structural_scores["sibling"] * 0.01
        )
        
        return min(1.0, base_score + definition_bonus + structural_bonus)
    
    weights = {
        "label": 0.50,
        "definition": 0.15,
        "parent": 0.15,
        "child": 0.12,
        "sibling": 0.08
    }
    
    # If no definition exists, redistribute that weight to label
    if definition_score == 0:
        weights["label"] += weights["definition"] * 0.5
        weights["parent"] += weights["definition"] * 0.2
        weights["child"] += weights["definition"] * 0.2
        weights["sibling"] += weights["definition"] * 0.1
        weights["definition"] = 0
    
    # If no structural context exists, give more weight to label and definition
    has_structure = any(structural_scores[k] > 0 for k in ["parent", "child", "sibling"])
    if not has_structure:
        structural_weight = weights["parent"] + weights["child"] + weights["sibling"]
        weights["label"] += structural_weight * 0.7
        weights["definition"] += structural_weight * 0.3
        weights["parent"] = weights["child"] = weights["sibling"] = 0
    
    combined_score = (
        weights["label"] * label_score +
        weights["definition"] * definition_score +
        weights["parent"] * structural_scores["parent"] +
        weights["child"] * structural_scores["child"] +
        weights["sibling"] * structural_scores["sibling"]
    )
    
    return combined_score

@app.get("/projects/{project_id}/suggestions_full")
def generate_node_suggestions_enhanced(
    project_id: int,
    node_uri: str = Query(..., description="URI of the selected node"),
    min_threshold: float = Query(0.4, description="Minimum similarity threshold"),
    structural_threshold: float = Query(0.65, description="Threshold for structural matching"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    
    project = db.query(Project).filter(
        Project.id == project_id, 
        Project.user_id == current_user.id
    ).first()
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

    node_uri_ref = URIRef(node_uri)
    
    # Extract all information about source node
    source_labels = get_all_labels(g1, node_uri_ref)
    if not source_labels:
        raise HTTPException(status_code=404, detail="Node not found in source ontology")
    
    source_definition = get_definition(g1, node_uri_ref)
    
    # Calculate suggestions
    suggestions = []
    seen_uris = set()
    
    for s2 in g2.subjects():
        if s2 in seen_uris:
            continue
        
        target_labels = get_all_labels(g2, s2)
        if not target_labels:
            continue
        
        # Calculate label similarity
        label_score, is_exact = calculate_label_similarity(source_labels, target_labels)
        
        # Calculate definition similarity
        target_definition = get_definition(g2, s2)
        definition_score = calculate_definition_similarity(
            source_definition, 
            target_definition
        )
        
        # Calculate structural similarity
        structural_scores = calculate_structural_similarity(
            g1, node_uri_ref, 
            g2, s2,
            structural_threshold
        )
        
        # Calculate combined score with adaptive weighting
        combined_score = calculate_combined_score(
            label_score,
            is_exact,
            definition_score,
            structural_scores
        )
        
        if combined_score >= min_threshold:
            suggestions.append({
                "node2": str(s2),
                "label2": target_labels[0],  # Primary label
                "all_labels": target_labels,
                "similarity": round(combined_score, 3),
                "is_exact_match": is_exact,
                "scores": {
                    "label": round(label_score, 3),
                    "definition": round(definition_score, 3),
                    "parent": round(structural_scores["parent"], 3),
                    "child": round(structural_scores["child"], 3),
                    "sibling": round(structural_scores["sibling"], 3)
                }
            })
            seen_uris.add(s2)
    
    # Sort by combined similarity (exact matches first, then by score)
    suggestions.sort(key=lambda x: (not x["is_exact_match"], -x["similarity"]))
    
    # Save results
    os.makedirs(SUGGESTIONS_DIR, exist_ok=True)
    output_file = os.path.join(SUGGESTIONS_DIR, f"{project_id}_enhanced.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(suggestions, f, ensure_ascii=False, indent=2)
    
    return {
        "node": node_uri,
        "source_labels": source_labels,
        "source_definition": source_definition,
        "suggestions": suggestions[:20],  # Return top 20
        "total_matches": len(suggestions),
        "exact_matches": sum(1 for s in suggestions if s["is_exact_match"])
    }

#---test skos tree node details
from typing import Dict, Any
def normalize_values(values):
    if isinstance(values, list):
        unique_values = list(set(values))
        return unique_values[0] if len(unique_values) == 1 else unique_values
    return values

@app.get("/node-details-skostree/")
def node_details_skostree(
    uri: str = Query(...),
    file_id: int = Query(...),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:

    file = db.query(File).filter(File.id == file_id).first()
    if not file:
        raise HTTPException(status_code=404, detail=f"File with id {file_id} not found")

    base_name = os.path.splitext(file.filename)[0]

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    nt_path = os.path.join(BASE_DIR, "uploads", "triples", f"{base_name}_triples.nt")

    if not os.path.exists(nt_path):
        nt_path = os.path.join("/app/uploads/triples", f"{base_name}_triples.nt")

    if not os.path.exists(nt_path):
        raise HTTPException(status_code=404, detail=f"NT file {nt_path} not found")

    # Ανάγνωση triples
    node_uri = URIRef(uri)
    details = {}
    found = False

    g = Graph()
    g.parse(nt_path, format="nt")
    for s, p, o in g.triples((node_uri, None, None)):
        pred = str(p).split("#")[-1] if "#" in str(p) else str(p).split("/")[-1]
        details.setdefault(pred, [])
        details[pred].append(str(o))
        found = True

    if not found:
        raise HTTPException(status_code=404, detail=f"Node {uri} not found in {file.filename}")

    details = {k: normalize_values(v) for k, v in details.items()}
    return {"uri": uri, "details": details}


@app.delete("/projects/{project_id}")
def delete_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=404, 
            detail="Project not found or you don't have permission to delete it"
        )
    
    links_deleted = db.query(Link).filter(Link.project_id == project_id).delete()
    
    if links_deleted > 0:
        db.query(Vote).filter(
            Vote.link_id.in_(
                db.query(Link.id).filter(Link.project_id == project_id)
            )
        ).delete(synchronize_session=False)
    
    project_name = project.name
    
    db.delete(project)
    db.commit()
    
    return {
        "message": f"Project '{project_name}' deleted successfully",
        "deleted_project_id": project_id,
        "deleted_links": links_deleted
    }

from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rdflib import Graph, URIRef
from rdflib.namespace import SKOS, RDFS
import json
import os
from fastapi import HTTPException, Depends, Query
from sqlalchemy.orm import Session

# Model configuration
MODEL_NAME = 'all-MiniLM-L6-v2'

embedding_model = None

def get_embedding_model():
    """Lazy load the sentence transformer model."""
    global embedding_model
    if embedding_model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        embedding_model = SentenceTransformer(MODEL_NAME)
        print("Model loaded successfully")
    return embedding_model

def get_all_labels(graph, uri_ref) -> List[str]:
    labels = []
    label_props = [SKOS.prefLabel, SKOS.altLabel, RDFS.label]
    
    for prop in label_props:
        for label in graph.objects(subject=uri_ref, predicate=prop):
            labels.append(str(label))
    
    return labels


def get_definition(graph, uri_ref) -> str:
    definition = graph.value(subject=uri_ref, predicate=SKOS.definition)
    return str(definition) if definition else ""


def get_parents(graph, uri_ref) -> Set[URIRef]:
    parents = set()
    for parent in graph.objects(subject=uri_ref, predicate=SKOS.broader):
        parents.add(parent)
    return parents


def get_children(graph, uri_ref) -> Set[URIRef]:
    children = set()
    for child in graph.objects(subject=uri_ref, predicate=SKOS.narrower):
        children.add(child)
    for child in graph.subjects(predicate=SKOS.broader, object=uri_ref):
        children.add(child)
    return children


def get_siblings(graph, uri_ref) -> Set[URIRef]:
    siblings = set()
    parents = get_parents(graph, uri_ref)
    
    for parent in parents:
        for sibling in get_children(graph, parent):
            if sibling != uri_ref:
                siblings.add(sibling)
    
    return siblings


def create_concept_text(labels: List[str], definition: str = "") -> str:
    
    text_parts = []
    
    if labels:
        text_parts.append(labels[0])
        
        if len(labels) > 1:
            alt_labels = ', '.join(labels[1:3])
            text_parts.append(f"Also known as: {alt_labels}")
    
    if definition:
        def_text = definition[:500] if len(definition) > 500 else definition
        text_parts.append(def_text)
    
    return ". ".join(text_parts)


def create_context_text(graph, uri_ref) -> str:
  
    context_parts = []
    
    parents = get_parents(graph, uri_ref)
    if parents:
        parent_labels = []
        for parent in list(parents)[:2]:
            p_labels = get_all_labels(graph, parent)
            if p_labels:
                parent_labels.append(p_labels[0])
        if parent_labels:
            context_parts.append(f"Parent categories: {', '.join(parent_labels)}")
    
    children = get_children(graph, uri_ref)
    if children and len(children) > 0:
        child_labels = []
        for child in list(children)[:3]:
            c_labels = get_all_labels(graph, child)
            if c_labels:
                child_labels.append(c_labels[0])
        if child_labels:
            context_parts.append(f"Includes: {', '.join(child_labels)}")
    
    return ". ".join(context_parts) if context_parts else ""



def calculate_label_similarity_embeddings(
    labels1: List[str],
    labels2: List[str],
    model: SentenceTransformer
) -> Tuple[float, bool]:

    if not labels1 or not labels2:
        return 0.0, False
    
    labels1_lower = [l.lower().strip() for l in labels1]
    labels2_lower = [l.lower().strip() for l in labels2]
    
    for l1 in labels1_lower:
        if l1 in labels2_lower:
            return 1.0, True
    
    all_labels = labels1 + labels2
    embeddings = model.encode(all_labels, convert_to_tensor=False, show_progress_bar=False)
    
    embeddings1 = embeddings[:len(labels1)]
    embeddings2 = embeddings[len(labels1):]
    
    similarities = cosine_similarity(embeddings1, embeddings2)
    max_similarity = float(np.max(similarities))
    
    return max_similarity, False


def calculate_concept_similarity_embeddings(
    text1: str,
    text2: str,
    model: SentenceTransformer
) -> float:
  
    if not text1 or not text2:
        return 0.0
    
    embeddings = model.encode([text1, text2], convert_to_tensor=False, show_progress_bar=False)
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    return float(similarity)


def calculate_definition_similarity_embeddings(
    def1: str,
    def2: str,
    model: SentenceTransformer
) -> float:
    if not def1 or not def2:
        return 0.0
    
    embeddings = model.encode([def1, def2], convert_to_tensor=False, show_progress_bar=False)
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    return float(similarity)


def calculate_structural_similarity_embeddings(
    graph1, uri1: URIRef,
    graph2, uri2: URIRef,
    model: SentenceTransformer,
    threshold: float = 0.70
) -> Dict[str, float]:
   
    parents1 = get_parents(graph1, uri1)
    parents2 = get_parents(graph2, uri2)
    children1 = get_children(graph1, uri1)
    children2 = get_children(graph2, uri2)
    siblings1 = get_siblings(graph1, uri1)
    siblings2 = get_siblings(graph2, uri2)
    
    def match_concept_sets(concepts1: Set[URIRef], concepts2: Set[URIRef]) -> float:
        if not concepts1 or not concepts2:
            return 0.0
        
        texts1 = []
        for c in concepts1:
            labels = get_all_labels(graph1, c)
            if labels:
                texts1.append(labels[0])
        
        texts2 = []
        for c in concepts2:
            labels = get_all_labels(graph2, c)
            if labels:
                texts2.append(labels[0])
        
        if not texts1 or not texts2:
            return 0.0
        
        embeddings1 = model.encode(texts1, convert_to_tensor=False, show_progress_bar=False)
        embeddings2 = model.encode(texts2, convert_to_tensor=False, show_progress_bar=False)
        
        # Calculate similarities
        similarities = cosine_similarity(embeddings1, embeddings2)
        
        # Count matches above threshold
        matches = 0
        for i in range(len(embeddings1)):
            if np.max(similarities[i]) >= threshold:
                matches += 1
        
        return matches / max(len(concepts1), len(concepts2))
    
    return {
        "parent": match_concept_sets(parents1, parents2),
        "child": match_concept_sets(children1, children2),
        "sibling": match_concept_sets(siblings1, siblings2)
    }


# ============================================================================
# Score Calculation
# ============================================================================

def calculate_combined_score_semantic(
    label_similarity: float,
    is_exact_match: bool,
    concept_similarity: float,
    definition_similarity: float,
    structural_scores: Dict[str, float],
    has_definition: bool
) -> float:

    if is_exact_match:
        base_score = 0.92
        
        concept_boost = concept_similarity * 0.05
        structural_avg = np.mean(list(structural_scores.values()))
        structural_boost = structural_avg * 0.03
        
        return min(1.0, base_score + concept_boost + structural_boost)
    
    weights = {
            "label": 0.20,
            "concept": 0.10,     
            "definition": 0.50,  
            "parent": 0.10,       
            "child": 0.05,        
            "sibling": 0.05       
        }
        
    if not has_definition:
            extra = weights["definition"]
            weights["concept"] += extra * 0.50    
            weights["label"] += extra * 0.30      
            weights["parent"] += extra * 0.15     
            weights["child"] += extra * 0.05
            weights["definition"] = 0
        
    has_structure = any(structural_scores[k] > 0 for k in ["parent", "child", "sibling"])
    if not has_structure:
            structural_weight = weights["parent"] + weights["child"] + weights["sibling"]
            weights["definition"] += structural_weight * 0.50
            weights["concept"] += structural_weight * 0.35     
            weights["label"] += structural_weight * 0.15
            weights["parent"] = weights["child"] = weights["sibling"] = 0
        
    combined = (
            weights["label"] * label_similarity +
            weights["concept"] * concept_similarity +
            weights["definition"] * definition_similarity +
            weights["parent"] * structural_scores["parent"] +
            weights["child"] * structural_scores["child"] +
            weights["sibling"] * structural_scores["sibling"]
        )
    return combined


@app.get("/projects/{project_id}/suggestions_semantic")
def generate_semantic_suggestions(
    project_id: int,
    node_uri: str = Query(..., description="URI of the selected node"),
    min_threshold: float = Query(0.5, description="Minimum similarity threshold (0.0-1.0)"),
    structural_threshold: float = Query(0.70, description="Threshold for structural matching"),
    top_k: int = Query(20, description="Number of top suggestions to return"),
    use_context: bool = Query(False, description="Include hierarchical context in embeddings"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    file1 = db.query(File).filter(File.id == project.file1_id).first()
    file2 = db.query(File).filter(File.id == project.file2_id).first()
    if not file1 or not file2:
        raise HTTPException(status_code=404, detail="Ontology files not found")
    
    format1 = guess_rdf_format(file1.resource)
    format2 = guess_rdf_format(file2.resource)
    g1, g2 = Graph(), Graph()
    g1.parse(file1.resource, format=format1)
    g2.parse(file2.resource, format=format2)
    
    model = get_embedding_model()
    
    node_uri_ref = URIRef(node_uri)
    
    source_labels = get_all_labels(g1, node_uri_ref)
    if not source_labels:
        raise HTTPException(status_code=404, detail="Node not found in source ontology")
    
    source_definition = get_definition(g1, node_uri_ref)
    source_concept_text = create_concept_text(source_labels, source_definition)
    
    if use_context:
        context = create_context_text(g1, node_uri_ref)
        if context:
            source_concept_text += ". " + context
    
    source_concept_embedding = model.encode(source_concept_text, convert_to_tensor=False, show_progress_bar=False)
    
    targets = []
    seen_uris = set()
    
    for s2 in g2.subjects():
        if s2 in seen_uris:
            continue
        
        target_labels = get_all_labels(g2, s2)
        if not target_labels:
            continue
        
        target_definition = get_definition(g2, s2)
        target_concept_text = create_concept_text(target_labels, target_definition)
        
        # Optionally include context
        if use_context:
            context = create_context_text(g2, s2)
            if context:
                target_concept_text += ". " + context
        
        targets.append({
            "uri": s2,
            "labels": target_labels,
            "definition": target_definition,
            "concept_text": target_concept_text
        })
        seen_uris.add(s2)
    
    print(f"Found {len(targets)} target concepts")
    
    suggestions = []
    
    if targets:
        target_texts = [t["concept_text"] for t in targets]
        print(f"Encoding {len(target_texts)} target concepts...")
        target_embeddings = model.encode(target_texts, convert_to_tensor=False, show_progress_bar=True)
        
        print("Calculating similarities...")
        for i, target in enumerate(targets):
            # Calculate label similarity
            label_sim, is_exact = calculate_label_similarity_embeddings(
                source_labels,
                target["labels"],
                model
            )
            
            # Calculate concept similarity (full semantic understanding)
            concept_sim = cosine_similarity(
                [source_concept_embedding],
                [target_embeddings[i]]
            )[0][0]
            
            # Calculate definition similarity if both exist
            def_sim = 0.0
            if source_definition and target["definition"]:
                def_sim = calculate_definition_similarity_embeddings(
                    source_definition,
                    target["definition"],
                    model
                )
            
            # Calculate structural similarity
            structural_scores = calculate_structural_similarity_embeddings(
                g1, node_uri_ref,
                g2, target["uri"],
                model,
                structural_threshold
            )
            
            combined_score = calculate_combined_score_semantic(
                label_sim,
                is_exact,
                concept_sim,
                def_sim,
                structural_scores,
                bool(source_definition and target["definition"])
            )
            
            if combined_score >= min_threshold:
                suggestions.append({
                    "node2": str(target["uri"]),
                    "label2": target["labels"][0],
                    "all_labels": target["labels"],
                    "definition": target["definition"],
                    "similarity": round(float(combined_score), 3),
                    "is_exact_match": is_exact,
                    "scores": {
                        "label": round(float(label_sim), 3),
                        "concept": round(float(concept_sim), 3),
                        "definition": round(float(def_sim), 3),
                        "parent": round(structural_scores["parent"], 3),
                        "child": round(structural_scores["child"], 3),
                        "sibling": round(structural_scores["sibling"], 3)
                    }
                })
    
    suggestions.sort(key=lambda x: (not x["is_exact_match"], -x["similarity"]))
    
    print(f"Found {len(suggestions)} matches above threshold {min_threshold}")
    
    os.makedirs(SUGGESTIONS_DIR, exist_ok=True)
    output_file = os.path.join(SUGGESTIONS_DIR, f"{project_id}_semantic.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(suggestions[:top_k], f, ensure_ascii=False, indent=2)
    
    return {
        "node": node_uri,
        "source_labels": source_labels,
        "source_definition": source_definition,
        "suggestions": suggestions[:top_k],
        "total_matches": len(suggestions),
        "exact_matches": sum(1 for s in suggestions if s["is_exact_match"]),
        "model_used": MODEL_NAME,
        "parameters": {
            "min_threshold": min_threshold,
            "structural_threshold": structural_threshold,
            "use_context": use_context
        }
    }