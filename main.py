from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, re, subprocess
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai
from google.genai import types

app = FastAPI(title="OpusClip Free Clone Backend")

# Allow CORS for netlify frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    youtube_url: str

def extract_video_id(url: str):
    reg_exp = r'^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*'
    match = re.search(reg_exp, url)
    return match.group(2) if match and len(match.group(2)) == 11 else None

@app.post("/api/analyze")
async def analyze_video(req: VideoRequest):
    video_id = extract_video_id(req.youtube_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    # 1. Fetch Transcripts
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([f"[{int(entry['start'])}s] {entry['text']}" for entry in transcript_list])
    except Exception as e:
        transcript_text = ""
        print(f"Transcript Error: {e}")

    # 2. Analyze with Gemini
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not configured.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are an expert short-form video editor. Locate the absolute most viral 30-60s window from this transcript.
    Video URL: {req.youtube_url}
    
    Transcript:
    {transcript_text[:12000]}
    """

    system_instruction = "Select the highest engagement starting time, ending time, title, and virality breakdown."

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "title": types.Schema(type=types.Type.STRING),
                        "description": types.Schema(type=types.Type.STRING),
                        "startTime": types.Schema(type=types.Type.INTEGER),
                        "endTime": types.Schema(type=types.Type.INTEGER),
                        "viralityScore": types.Schema(type=types.Type.INTEGER),
                        "hookScore": types.Schema(type=types.Type.INTEGER),
                        "retentionScore": types.Schema(type=types.Type.INTEGER),
                        "clickabilityScore": types.Schema(type=types.Type.INTEGER),
                        "reasons": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                        "hashtags": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                        "clipTranscript": types.Schema(type=types.Type.STRING)
                    },
                    required=["title", "description", "startTime", "endTime", "viralityScore", "hookScore", "retentionScore", "clickabilityScore", "reasons", "hashtags", "clipTranscript"]
                )
            )
        )
        return {"video_id": video_id, "analysis": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini processing error: {str(e)}")

# CLI Trimming Command Endpoint inside Render (requires ffmpeg and yt-dlp installed)
# Instructs FFmpeg to trim the video and crop to 9:16 vertical using best-center extraction.
@app.get("/api/download-short")
def cut_short(video_id: str, start_time: int, end_time: int):
    os.makedirs("/tmp/shorts", exist_ok=True)
    raw_video_path = f"/tmp/{video_id}_raw.mp4"
    output_short_path = f"/tmp/shorts/{video_id}_short.mp4"

    # Download YouTube stream stably via automated yt-dlp CLI
    download_cmd = [
        "yt-dlp", "-f", "best[ext=mp4]", "--force-overwrites", "-o", raw_video_path,
        f"https://www.youtube.com/watch?v={video_id}"
    ]
    try:
        subprocess.run(download_cmd, check=True)
    except Exception as d_err:
        raise HTTPException(status_code=500, detail=f"Failed downloading stream: {str(d_err)}")

    # Trim and extract center 9:16 coordinates
    duration = end_time - start_time
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-ss", str(start_time), "-i", raw_video_path, "-t", str(duration),
        "-vf", "crop=ih*9/16:ih,scale=1080:1920",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
        output_short_path
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        if os.path.exists(raw_video_path):
            os.remove(raw_video_path)
    except Exception as f_err:
        raise HTTPException(status_code=500, detail=f"FFmpeg crop error: {str(f_err)}")

    from fastapi.responses import FileResponse
    return FileResponse(output_short_path, media_type="video/mp4", filename=f"short_{video_id}.mp4")
