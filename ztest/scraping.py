import requests
from bs4 import BeautifulSoup

url = "https://agmarknet.gov.in"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract relevant data (modify as needed)
data = soup.find_all('table')[0]  # Example table selection
print(data)
