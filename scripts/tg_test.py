import os                                                           # access environment variables
import requests                                                     # send web requests
from dotenv import load_dotenv                                      # read the .env file

load_dotenv()                                                       # load TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from .env
r = requests.post(                                                  # send a POST request to the Telegram Bot API
    f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage",   # the sendMessage address, with our token in it
    data={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": "test from nextranet project"},   # who receives it, and the text
    timeout=15)                                                     # give up after 15 seconds (so a blocked network does not hang)
print(r.status_code, r.text)                                        # 200 = sent; anything else shows the error
