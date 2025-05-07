import os 
from dotenv import load_dotenv
from totalsegmentator.python_api import totalsegmentator

# Load environment variables from .env file
load_dotenv()
os.environ["TOTALSEG_WEIGHTS_PATH"] = os.getenv("TOTALSEG_WEIGHTS_PATH")
print(f"Loading Model From: {os.getenv('TOTALSEG_WEIGHTS_PATH')}" )

totalsegmentator("/data/RSNAPE/test/ff62ec60c99b/0e5fa221590c", output="example", preview=True, task="Total")
