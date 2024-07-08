from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

temperature = 33.76
humidity = 0.98
aqi = 80

input_prompt = f"""
Berikut adalah data dari sensor ESP32
temperature: {temperature} 
humidity: {humidity} 
aqi: {aqi} 

Berdasarkan data diatas, buatlah beberapa saran untuk pengguna:"""

result = llm.invoke(input_prompt)
print(result.content)
