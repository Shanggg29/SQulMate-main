from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
import subprocess

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Temporary folder to save uploaded lecture files
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------
# Routes for pages
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("app.html", {"request": request})

@app.get("/app", response_class=HTMLResponse)
def app_page(request: Request):
    return templates.TemplateResponse("app.html", {"request": request})

@app.get("/summary", response_class=HTMLResponse)
def summary_page(request: Request):
    return templates.TemplateResponse("summary.html", {"request": request})

@app.get("/qa", response_class=HTMLResponse)
def qa_page(request: Request):
    return templates.TemplateResponse("qa.html", {"request": request})

# -------------------------------
# Helper functions
# -------------------------------
def get_latest_extractive_path():
    path = os.path.join(UPLOAD_FOLDER, "extractive_summary.txt")
    return path if os.path.exists(path) else None

def get_latest_essay_path():
    path = os.path.join(UPLOAD_FOLDER, "summary_essays.txt")
    return path if os.path.exists(path) else None

def get_latest_ai_extracted_path():
    path = os.path.join(UPLOAD_FOLDER, "ai_extracted.txt")
    return path if os.path.exists(path) else None

def get_latest_qa_path():
    path = os.path.join(UPLOAD_FOLDER, "qa_output.txt")
    return path if os.path.exists(path) else None

# -------------------------------
# Serve the extractive / essay summary
# -------------------------------
@app.get("/extractive_summary")
def get_extractive_summary():
    path = get_latest_extractive_path()
    if path:
        return FileResponse(path, media_type="text/plain")
    return {"error": "Extractive summary not found"}

@app.get("/extractive_summary_essay")
def get_extractive_summary_essay():
    path = get_latest_essay_path()
    if path:
        return FileResponse(path, media_type="text/plain")
    return {"error": "Essay summary not found"}

# -------------------------------
# Process lecture (file OR manual text)
# -------------------------------
@app.post("/process_lecture")
async def process_lecture(
    lecture_file: UploadFile = File(None),
    manual_text: str = Form(None)
):
    try:
        # File uploaded
        if lecture_file:
            allowed_exts = ["pdf", "txt", "ppt", "pptx", "doc", "docx"]
            ext = lecture_file.filename.split(".")[-1].lower()
            if ext not in allowed_exts:
                return JSONResponse({"success": False, "error": "Invalid file type."})
            temp_path = os.path.join(UPLOAD_FOLDER, lecture_file.filename)
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(lecture_file.file, buffer)
            input_path = temp_path

        # Manual text
        elif manual_text:
            temp_path = os.path.join(UPLOAD_FOLDER, "manual_input.txt")
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(manual_text)
            input_path = temp_path

        else:
            return JSONResponse({"success": False, "error": "No file or manual text provided."})

        # --- Run newversion.py ---
        newversion_path = r"C:\Users\HP-USER\Downloads\SQulMate\summary\newversion.py"
        
        if manual_text:
            # For manual text, use --text flag
            subprocess.run(["python", newversion_path, "--text", manual_text], check=True, cwd=UPLOAD_FOLDER)
        else:
            # For file upload, pass file path
            subprocess.run(["python", newversion_path, input_path], check=True)

        # --- Run summarizing_run.py ---
        ai_extracted_path = os.path.join(UPLOAD_FOLDER, "ai_extracted.txt")
        summarizing_run_path = r"C:\Users\HP-USER\Downloads\SQulMate\summary\summarizing_run.py"
        subprocess.run(["python", summarizing_run_path, ai_extracted_path], check=True)

        # --- Read final extractive summary ---
        extractive_summary_path = get_latest_extractive_path()
        if not extractive_summary_path:
            return JSONResponse({"success": False, "error": "Extractive summary not found."})

        with open(extractive_summary_path, "r", encoding="utf-8") as f:
            summary_text = f.read()

        return JSONResponse({"success": True, "summary": summary_text})

    except subprocess.CalledProcessError as e:
        return JSONResponse({"success": False, "error": f"Error running script: {str(e)}"})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})

# -------------------------------
# Generate Q&A endpoint
# -------------------------------
@app.post("/generate_qna")
async def generate_qna():
    try:
        ai_extracted_path = get_latest_ai_extracted_path()
        if not ai_extracted_path:
            return JSONResponse({"success": False, "error": "ai_extracted.txt not found."})

        qna_path = r"C:\Users\HP-USER\Downloads\SQulMate\question\qna.py"
        subprocess.run(["python", qna_path, ai_extracted_path], check=True)

        qa_output_path = get_latest_qa_path()
        if not qa_output_path:
            return JSONResponse({"success": False, "error": "Q&A output not found."})

        with open(qa_output_path, "r", encoding="utf-8") as f:
            qa_text = f.read()

        return JSONResponse({"success": True, "qa": qa_text})

    except subprocess.CalledProcessError as e:
        return JSONResponse({"success": False, "error": f"Error running qna.py: {str(e)}"})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})