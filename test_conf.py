import requests

with open("lena.jpg", "rb") as f:
    res = requests.post("http://localhost:5000/api/register", files={"image": f}, data={"name": "Lena"})

with open("lena.jpg", "rb") as f:
    res = requests.post("http://localhost:5000/api/recognize", files={"image": f})
    data = res.json()
    print("Match confidence:", data['faces'][0]['confidence'])

res = requests.delete("http://localhost:5000/api/faces/Lena")
