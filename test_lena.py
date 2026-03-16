import requests
with open("lena.jpg", "rb") as f:
    res = requests.post("http://localhost:5000/api/recognize", files={"image": f})
print(res.json())
