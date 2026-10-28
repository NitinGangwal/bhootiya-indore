# 🕯️ Bhootiya Indore: Dadi-Nani ki Kahaniyan

Ghar ke bade logon se suni hui Indore ki bhootiya kahaniyan, jo ek offline "ofrenda" mein sambhal ke rakhi jaati hain.

Hacktoberfest Open-Source AI Challenge Week 4 ("Ghost in the Machine") ke liye banaya gaya.

## Kaise kaam karta hai
1. Dadi/Nani ki awaaz record karo (phone ya WhatsApp voice note)
2. **faster-whisper** (open-source) audio ko local machine par Hindi text mein badalta hai
3. **Gemma** (open-weight, Ollama ke through local) kahani se title, jagah, summary, darr level aur eerie opening line nikaalta hai
4. **Streamlit** mein sab kahaniyan spooky ofrenda cards ban jaati hain, asli awaaz ke saath

## Open-source kyun?
- Family ki awaaz aur kahaniyan kisi server par nahi jaatin
- Internet ke bina chalta hai
- Koi service band ho jaye, kahaniyan phir bhi apni machine par safe

## Chalane ka tareeka
```bash
# 1. Ollama install karo (https://ollama.com), phir:
ollama pull gemma3:4b

# 2. Python packages
pip install -r requirements.txt

# 3. App chalao
streamlit run app.py
```

## Tech stack
Python · faster-whisper · Gemma (via Ollama) · Streamlit

## Licence
MIT
