import requests
import urllib.parse

try:
    # First get all faces
    res = requests.get('http://localhost:5000/api/faces')
    faces = res.json()
    print("Faces in DB:", faces)
    
    if faces.get('success') and faces['people']:
        name = faces['people'][0]['name']
        print(f"Trying to delete '{name}'...")
        encoded_name = urllib.parse.quote(name)
        del_res = requests.delete(f'http://localhost:5000/api/faces/{encoded_name}')
        print("Delete Response:", del_res.status_code, del_res.json())
        
        # Verify
        res2 = requests.get('http://localhost:5000/api/faces')
        print("Faces in DB after:", res2.json())
    else:
        print("No faces to delete.")
except Exception as e:
    print("Error:", e)
