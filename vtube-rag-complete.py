import os
import time
import sys
import re
from googleapiclient.discovery import build
import google.generativeai as genai
from gtts import gTTS
from pydub import AudioSegment
import pyaudio
import pickle
from langchain.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List
import logging
import time
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API Keys
YOUTUBE_API_KEY = 'AIzaSyBuuksoZSdSpga7JtEqfEw3ivfHwISwEMA'
GEMINI_API_KEY = 'AIzaSyDtJc_rKh1iQLHL6xuZjZWBesbvREGwagY'

SCREENED_WORDS = ["วิชาบังคับ", "วิชาเฉพาะ"]
DEFINE_WORDS = {
    "วิชาบังคับ": "หมายถึง วิชาบังคับไม่เกี่ยวกับกลุ่มวิชาเฉพาะ",
    "วิชาเฉพาะ": "หมายถึง วิชาเฉพาะซึ่งแยกจากวิชาบังคับ",
    "หมวดวิชาอะไร": "หมายถึง หมวดวิชาศึกษาทั่วไป และ หมวดวิชาเฉพาะ",
}

def define_words(word):
    """ให้ความหมายของคำ"""
    if word in DEFINE_WORDS:
        return DEFINE_WORDS[word]
    return f"ไม่พบความหมายของคำว่า {word}"

# หาตำแหน่ง PDF แล้วก็ โหลดข้อมูลPDF ทุกหน้าแล้วเอา Gemini มาวิเคราะห์ข้อมูล
def initialize_ai(pdf_paths: List[str]):
    try:
        genai.configure(api_key=GEMINI_API_KEY)

        all_pages = []
        for pdf_path in pdf_paths:
            logging.info(f"กำลังโหลด {pdf_path}...")
            try:
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                all_pages.extend(pages)
            except Exception as pdf_load_error:
                logging.error(f"Error loading PDF {pdf_path}: {pdf_load_error}")

        if not all_pages:
            raise ValueError("ไม่มีหน้าในไฟล์ PDF ที่โหลดได้")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=9000,  # Adjust chunk size
            chunk_overlap=900, # Adjust overlap
            separators=["\n\n", "\n", ". ", " ", ""], # Add empty string as separator
            length_function=len,
        )
        chunks = splitter.split_documents(all_pages)
        logging.info(f"จำนวน chunks ที่สร้าง: {len(chunks)}")

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        logging.info("Vector store ถูกสร้างเรียบร้อย")

        return vector_store
    except Exception as e:
        logging.error(f"Error initializing AI: {str(e)}")
        return None


# ติดตั้งตัว Api Youtube และเชื่่อมแชทไลฟ์สด
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY) 
def get_live_chat_id(channel_id):
    try:
        search_request = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            eventType="live",
            type="video"
        )
        search_response = search_request.execute()

        if "items" in search_response and search_response["items"]:
            video_id = search_response["items"][0]["id"]["videoId"]
            video_request = youtube.videos().list(
                part="liveStreamingDetails",
                id=video_id
            )
            video_response = video_request.execute()

            if "items" in video_response and video_response["items"]:
                live_chat_id = video_response["items"][0]["liveStreamingDetails"].get("activeLiveChatId")
                if live_chat_id:
                    print(f"พบ Live Chat ID: {live_chat_id}")
                    return live_chat_id

        print("ไม่พบการไลฟ์สดที่กำลังดำเนินอยู่")
        return None
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการดึง live chat ID: {e}")
        return None


# ดึงข้อความในแชทไลฟ์สด
def fetch_live_chat_messages(live_chat_id, next_page_token=None):
    try:
        request = youtube.liveChatMessages().list(
            liveChatId=live_chat_id,
            part="snippet,authorDetails",
            pageToken=next_page_token
        )
        response = request.execute()
        messages = [
            {
                "author": item["authorDetails"]["displayName"],
                "message": item["snippet"].get("textMessageDetails", {}).get("messageText", "")
            }
            for item in response.get("items", [])
        ]
        next_page_token = response.get("nextPageToken")
        return messages, next_page_token
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการดึงข้อความ: {e}")
        return [], None

def send_live_chat_message(live_chat_id, message_text):
    try:
        youtube.liveChatMessages().insert(
            part="snippet",
            body={
                "snippet": {
                    "liveChatId": live_chat_id,
                    "type": "textMessageEvent",
                    "textMessageDetails": {
                        "messageText": message_text
                    }
                }
            }
        ).execute()
    except Exception as e:
       return None

# ทำให้บอทพูดจากข้อความ
def text_to_speech(text, language="th", tld='co.th', slow=False ,lang_check=True,remove_asterisk=True,symbols_to_keep=".,?!", pitch_shift=0, speed_factor=1.2):
    
    pattern = ""
    if remove_asterisk:
        pattern += r"\*"  # ลบดอกจัน
    else:
        escaped_symbols = re.escape(symbols_to_keep)
        pattern += f"[^a-zA-Z0-9\s{escaped_symbols}]"
    cleaned_text = re.sub(pattern, "", text)

    try:
        
        tts = gTTS(text=cleaned_text, lang=language, tld=tld, slow=slow, lang_check=lang_check)
        tts.save("output.mp3")
        sound = AudioSegment.from_mp3("output.mp3")

        pitch_frame_rate = int(sound.frame_rate * (2.0 ** (pitch_shift / 12)))
        pitched_sound = sound._spawn(sound.raw_data, overrides={"frame_rate": pitch_frame_rate})
        speed_frame_rate = int(pitched_sound.frame_rate * speed_factor)
        adjusted_sound = pitched_sound._spawn(pitched_sound.raw_data, overrides={"frame_rate": speed_frame_rate})

        adjusted_sound = adjusted_sound.set_frame_rate(44100)
        adjusted_sound.export("output_adjusted.wav", format="wav")

        p = pyaudio.PyAudio()
        device_index = None
        target_device_name = "CABLE Input" 

        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if target_device_name in info.get("name", ""):
                device_index = i
                break

        if device_index is None:
            raise RuntimeError("ไม่พบ CABLE Input")

        stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100,
                        output=True, output_device_index=device_index)

        with open("output_adjusted.wav", "rb") as wf:
            data = wf.read(1024)
            while data:
                stream.write(data)
                data = wf.read(1024)

        stream.stop_stream()
        stream.close()
        p.terminate()

    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการแปลงเสียง: {e}")
    finally:
        # Clean up temporary files
        if os.path.exists("output.mp3"):
            os.remove("output.mp3")
        if os.path.exists("output_adjusted.wav"):
            os.remove("output_adjusted.wav")

# ลบดอกจันจากข้อความ แต่ถ้าเอาออกไม่ได้ถือว่าเป้่นการสำรองลบดอกจันจากเสียง
def remove_unwanted_markers(text):
    """Removes unwanted markers from a given text using various methods.

    Args:
        text: The input text string.

    Returns:
        The cleaned text string.
    """
    text = re.sub(r"\[เอกสารแนบ]|(\*\*)|(\*+)|(^\*+|\*+$)|(\s*\*+\s*)|\s+|\<[^>]+>|[\u200B-\u200D\uFEFF]|[\x00-\x1F\x7F]", " ", text).strip()
    return text

def play_waiting_sound_loop(stop_event, message="กรุณารอสักครู่นะคะ"):
    """Plays the waiting sound in a loop until the stop_event is set."""
    logging.info(f"Waiting sound loop started with message: '{message}'")
    while not stop_event.is_set():
        if stop_event.is_set(): 
            break
        text_to_speech(message)
        time.sleep(0.2) 
    logging.info("Waiting sound loop stopped.")

# ตั้งค่าให้ Gemini สอดคล้องกับเอกสารที่มีมากที่สุด
def get_rag_response(vector_store, question, max_retries=3):
    for attempt in range(max_retries):
        try:
            docs = vector_store.similarity_search(question, k=30) 
            contexts = []
            for doc in docs:
                content = doc.page_content
                content = remove_unwanted_markers(content)
                contexts.append(content)
            context = "\n".join(contexts)

            model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
            prompt = f"""คุณเป็นผู้ช่วยอัจฉริยะที่เชี่ยวชาญในการให้ข้อมูลจากเอกสารที่ให้มาเท่านั้น ตอบคำถามต่อไปนี้โดยใช้ข้อมูลที่ให้มาเท่านั้น ห้ามสร้างข้อมูลขึ้นเอง หรืออ้างอิงแหล่งข้อมูลภายนอกใดๆ ทั้งสิ้น หากไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร ให้ตอบว่า "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร"
            - ให้ข้อมูลที่ถูกต้องและแม่นยำที่สุดจากเอกสารเท่านั้น
            - สรุปใจความสำคัญของข้อมูลที่เกี่ยวข้องกับคำถามโดยตรง
            - ใช้ภาษาที่เป็นธรรมชาติและกระชับ
            - หลีกเลี่ยงการใช้คำฟุ่มเฟือย
            - หากคำถามกำกวม ให้พยายามตีความหมายที่ดีที่สุดจากบริบทของเอกสาร
            - แยกคำให้ชัดเจน
            - หากมีข้อมูลหลายส่วนที่เกี่ยวข้อง ให้สรุปและเชื่อมโยงข้อมูลเหล่านั้นเข้าด้วยกันอย่างเหมาะสม
            - ทบทวนคำถามโดยละเอียด เช่น วิชาเฉพาะ กับ วิชาเฉพาะเลือก 
            ข้อมูล: {context}
            คำถาม: {question}"""

            print("กำลังคิด...")
            threading.Thread(target=lambda: text_to_speech("กรุณารอสักครู่นะคะ")).start()
            response = model.generate_content(prompt)
            answer = response.text.strip()

            if len(answer.split()) < 50 or '[เอกสารแนบ]' in answer:
                retry_prompt = f"""ทบทวนคำถาม: {question} โดยใช้ข้อมูลนี้: {context} {prompt}
                \n\nโปรดให้คำตอบที่ตรงตามเอกสาร โดยยังคงยึดตามคำแนะนำเดิม""" # ใช้ prompt เดิมเป็นฐาน
                print("กำลังทบทวนคำตอบ...")
                retry_response = model.generate_content(retry_prompt)
                answer = retry_response.text.strip()

            return answer
    
        except Exception as e:
            logging.error(f"เกิดข้อผิดพลาดในการตอบคำถาม (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"ขออภัยค่ะ เกิดข้อผิดพลาดร้ายแรงในการตอบคำถาม"
        

# เชื่อมต่อไลฟ์แชท
def process_live_chat(channel_id, pdf_paths: List[str]):
    print("กำลังเริ่มต้นระบบ...")
    vector_store = initialize_ai(pdf_paths) 

    if vector_store is None:
        print("ไม่สามารถเริ่มต้นระบบได้")
        return

    live_chat_id = get_live_chat_id(channel_id)
    if not live_chat_id:
        return

    print("เชื่อมต่อกับแชทสำเร็จ!")
    next_page_token = None
    processed_messages = set()

    while True:
        try:
            messages, next_page_token = fetch_live_chat_messages(live_chat_id, next_page_token)

            for message in messages:
                message_id = f"{message['author']}:{message['message']}"

                if not message["message"] or message_id in processed_messages:
                    continue

                processed_messages.add(message_id)
                print(f"{message['author']}: {message['message']}")

                try:
                    response = get_rag_response(vector_store, message["message"])
                    print(f"Bot: {response}")
                    send_live_chat_message(live_chat_id, response)
                    text_to_speech(response)
                except Exception as message_process_error: 
                    print(f"เกิดข้อผิดพลาดในการประมวลผลข้อความ: {message_process_error}")

            time.sleep(5)

            if len(processed_messages) > 1000:
                processed_messages = set(list(processed_messages)[-500:])

        except Exception as chat_process_error: 
            print(f"เกิดข้อผิดพลาดในการประมวลผลแชท: {chat_process_error}")
            time.sleep(5)

if __name__ == "__main__":
    try:
        channel_id = "UC01KD3H6ymneOqVIP4bCPew" #รหัสช่อง
        pdf_files = [  
            "control.pdf",
            "speci.pdf",
            "grand.pdf",
            "luck.pdf",
            "last.pdf",
            "term.pdf",
        ]

        # ตรวจสอบว่ามีไฟล์ PDF อยู่จริง
        valid_pdf_files = []
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                valid_pdf_files.append(pdf_file)
            else:
                print(f"คำเตือน: ไม่พบไฟล์ PDF: {pdf_file}")

        if not valid_pdf_files:
            raise FileNotFoundError("ไม่พบไฟล์ PDF ที่ถูกต้อง")

        if not channel_id:
            sys.exit("Error: Missing YouTube channel ID")

        print("กำลังเริ่มต้น VTube Bot...")
        process_live_chat(channel_id, valid_pdf_files) 
    except KeyboardInterrupt:
        print("\nปิดระบบ...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)