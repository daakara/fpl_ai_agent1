import os
from dotenv import load_dotenv

load_dotenv()  # loads environment variables from a .env file if present

FPL_USER_ID = int(os.getenv("FPL_USER_ID", "1437677"))
FPL_BASE_URL = "https://fantasy.premierleague.com/api/"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_NUMBER_ID = os.getenv("WHATSAPP_NUMBER_ID", "")
