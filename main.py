from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from uuid import uuid4
from datetime import datetime, timedelta
import shutil
import os
import json

# AI + OCR Tools
from openai import OpenAI
from dotenv import load_dotenv
import pytesseract
from PIL import Image
import pdfplumber

from PIL import ImageOps

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

with open("application_requirements.json", "r") as f:
    application_requirements = json.load(f)

#get specific API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()
sessions = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_json_file(filename):
    with open(filename, "r") as f:
        return json.load(f)

document_types = load_json_file("document_types.json")
application_types = load_json_file("application_types.json")

from fastapi import Query

@app.get("/get-required-docs")
def get_required_docs(application_type: str = Query(...)):
    if application_type not in application_requirements:
        raise HTTPException(status_code=404, detail="Application type not found")

    return {
        "application_type": application_type,
        "display": application_requirements[application_type]["display"],
        "required_categories": application_requirements[application_type]["required_categories"]
    }

@app.get("/required-docs/{application_type}")
def get_required_docs(application_type: str):
    if application_type not in application_types:
        raise HTTPException(status_code=404, detail="Unknown application type")

    ai_query = application_types[application_type]["ai_query"]

    # Ask OpenAI what documents are needed
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an immigration document assistant. Your job is to tell users what documents are typically required for a given application."},
            {"role": "user", "content": ai_query}
        ],
        temperature=0.3
    )

    text = response.choices[0].message.content.lower()
    matched_docs = []

    # Check what known documents were mentioned
    for doc_type, doc_info in document_types.items():
        if doc_info["display"].lower() in text or doc_type in text:
            matched_docs.append({
                "name": doc_type,
                "display": doc_info["display"],
                "fields": doc_info["fields"],
                "redact_fields": doc_info["redact_fields"]
            })

    if not matched_docs:
        return {"warning": "No matching known documents found in AI response", "ai_response": text}

    return {"documents": matched_docs}

@app.get("/get-document-type")
def get_document_type(doc_type: str):
    with open("document_types.json", "r") as f:
        document_types = json.load(f)

    if doc_type not in document_types:
        raise HTTPException(status_code=404, detail="Document type not found")

    return {
        "doc_type": doc_type,
        "fields": document_types[doc_type].get("fields", [])
    }


@app.post("/start-session")
def start_session():
    session_id = str(uuid4())
    expiry = datetime.utcnow() + timedelta(hours=3)
    sessions[session_id] = {"created": datetime.utcnow(), "expires": expiry, "files": []}
    return {"session_id": session_id, "expiry": expiry.isoformat()}

@app.post("/upload-doc")
def upload_doc(session_id: str = Form(...), file: UploadFile = None):
    if session_id not in sessions:
        sessions[session_id] = {
            "redacted_files": {},
            "form_data": {},
            "created_at": datetime.utcnow()
        }

    path = f"uploads/{session_id}/"
    os.makedirs(path, exist_ok=True)
    with open(path + file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    sessions[session_id]['files'].append(path + file.filename)
    return {"msg": "Uploaded"}

@app.get("/requirements")
def get_requirements():
    with open("doc_requirements.json") as f:
        data = json.load(f)
    return data

@app.post("/end-session")
def end_session(session_id: str = Form(...)):
    if session_id in sessions:
        shutil.rmtree(f"uploads/{session_id}", ignore_errors=True)
        del sessions[session_id]
    return {"msg": "Session ended and data deleted"}

    #analyze docs endpoint
@app.post("/analyze-docs")
def analyze_documents(session_id: str = Form(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    results = []
    for file_path in sessions[session_id]['files']:
        raw_text = extract_text_from_file(file_path)
        key_info = extract_key_info(raw_text)
        results.append({"file": os.path.basename(file_path), "info": key_info})
    
    return {"results": results}

@app.post("/chat-followup")
async def chat_followup(payload: dict):
    history = payload.get("history", [])

    messages = [{"role": "system", "content": "You are an AI that helps users fix mistakes in government document applications."}]

    for i, msg in enumerate(history):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": msg})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3
    )

    reply = response.choices[0].message.content
    return {"reply": reply}

@app.post("/upload-redacted-doc")
async def upload_redacted_doc(
    session_id: str = Form(...),
    document_name: str = Form(...),
    file: UploadFile = File(...)
):
    # 🛡️ Ensure session exists
    if session_id not in sessions:
        sessions[session_id] = {
            "redacted_files": {},
            "form_data": {},
            "created_at": datetime.utcnow()
        }

    # 📂 Create upload folder for session
    session_dir = os.path.join("uploads", session_id)
    os.makedirs(session_dir, exist_ok=True)

    # 📦 Create filename and save file
    filename = f"{document_name}_{uuid4().hex}_{file.filename.replace(' ', '_')}"
    file_path = os.path.join(session_dir, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🧠 Track redacted file path under session
    sessions[session_id]["redacted_files"][document_name] = file_path

    print(f"✅ Redacted file saved: {file_path}")
    print(f"📌 sessions[{session_id}]['redacted_files'] = {sessions[session_id]['redacted_files']}")

    return JSONResponse(content={
        "message": f"Redacted {document_name} saved",
        "file_path": file_path
    })

@app.post("/submit-application")
async def submit_application(session_id: str = Form(...), form_data: str = Form("{}")):
    if session_id not in sessions:
        sessions[session_id] = {
            "redacted_files": {},
            "form_data": {},
            "created_at": datetime.utcnow()
        }

    print("📦 Redacted files for submission:", sessions[session_id].get("redacted_files", {}))

    # 🧠 Collect redacted docs and manual fields
    redacted_files = sessions[session_id].get("redacted_files", {})
    manual_fields = json.loads(form_data)

    #DEBUG DL 
    print("🔍 redacted_files:", redacted_files)
    print("🔍 form_data raw:", form_data)
    print("🔍 parsed manual_fields:", manual_fields)
    #print(f"🔍 Processing redacted file: {path}")


    if not redacted_files and not manual_fields:
        raise HTTPException(status_code=400, detail="No input provided")

    # 📖 Extract text from all redacted files
    extracted_data = {}
    for doc_name, path in redacted_files.items():
        text = ""
        try:
            if path.endswith(".pdf"):
                with pdfplumber.open(path) as pdf:
                    text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                    print(f"🔍 Extracted from {path}: {text[:300]}") #DEBBUGGIN DL


            elif path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg"):
                print(f"🧠 Running OCR on image: {path}")
                try:
                    img = Image.open(path)

                    # 🔧 Convert to grayscale (improves OCR accuracy)
                    img = ImageOps.grayscale(img)

                    # 🧼 Optional: increase contrast / sharpness if needed
                    # img = img.point(lambda x: 0 if x < 150 else 255, '1')  # binarize (optional)

                    # 🧠 Run OCR
                    text = pytesseract.image_to_string(img)
                    print(f"📄 OCR extracted from {doc_name}:\n{text[:500]}")

                except Exception as e:
                    print(f"❌ OCR failed for {doc_name}: {e}")
                    text = "[Error running OCR]"

            else:
                    text = "[Unsupported file type]"
        except Exception as e:
            text = f"[Error reading file: {e}]"
        
        extracted_data[doc_name] = text

    # Combine everything for the AI prompt
    application_type = sessions[session_id].get("application_type", "ssn_replacement")  # or pass explicitly
    application_info = application_requirements.get(application_type, {})
    app_prompt = application_info.get("ai_prompt", "")

    prompt_parts = [
        "You are an immigration document assistant. Help verify this user's application."
    ]

    if app_prompt:
        prompt_parts.append(f"📌 Application Context: {app_prompt}")

    prompt_parts.append("Extracted document text:")
    for doc, content in extracted_data.items():
        print("FOUND 1 DOC🧠🧠")
        prompt_parts.append(f"📄 {doc}:\n{content}\n")

    if manual_fields:
        prompt_parts.append("Manually entered fields:")
        for doc, fields in manual_fields.items():
            pretty_fields = "\n".join([f"{k}: {v}" for k, v in fields.items()])
            prompt_parts.append(f"📝 {doc}:\n{pretty_fields}")

    prompt_parts.append("Check for missing documents, expired items, or mismatches. Suggest alternatives if needed.")

    final_prompt = "\n\n".join(prompt_parts)

    # 🧠 Run OpenAI validation
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in verifying government application packages."},
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.3
    )

    reply = response.choices[0].message.content

    return JSONResponse(content={
        "status": "success",
        "analysis": reply
    })

#functions
def cleanup_expired_sessions():
    now = datetime.utcnow()
    for sid in list(sessions.keys()):
        created = sessions[sid].get("created_at")
        if created and (now - created).total_seconds() > 10800:  # 3 hours
            del sessions[sid]
            print(f"🧹 Deleted expired session: {sid}")

    #extract text
def extract_text_from_file(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
        return pytesseract.image_to_string(Image.open(file_path))
    else:
        return ""

    #openAI prompt for info extraction
def extract_key_info(raw_text: str) -> dict:
    prompt = f"""
You are a document processing AI. Extract the following fields from the given text:

- Full Name
- Date of Birth
- Document Type (e.g., SSN, I-94, I-20)
- Document ID (if available)
- Expiration Date (if available)

Respond ONLY with strict JSON, like this:
{{
  "full_name": "...",
  "dob": "...",
  "document_type": "...",
  "document_id": "...",
  "expiration_date": "..."
}}

Text:
\"\"\"
{raw_text}
\"\"\"
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful document analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    raw_output = response.choices[0].message.content
    print("AI Output:\n", raw_output)

    import json
    try:
        return json.loads(raw_output)
    except Exception as e:
        print("JSON Parse Error:", str(e))
        return {"error": "Failed to parse response from OpenAI"}
