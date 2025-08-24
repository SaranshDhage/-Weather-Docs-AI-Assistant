# import requests

# API_KEY = "820d94dc3125280fc7f20e51a9cc0c06"
# CITY = "mumbai"
# URL = "https://api.openweathermap.org/data/2.5/weather"

# params = {
#     "q": CITY,
#     "appid": API_KEY,
#     "units": "metric"
# }

# try:
#     response = requests.get(URL, params=params, timeout=10)
#     response.raise_for_status()
#     data = response.json()
#     print("✅ Success! Weather data received:")
#     print(data)
# except requests.exceptions.RequestException as e:
#     print("❌ Error:", e)


from qdrant_client import QdrantClient

# 🔧 Replace with your actual values
QDRANT_URL = "https://d92626c2-1f04-4ba3-a3d6-199f5591a1ef.eu-west-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.QslcGgKv_1T7h28hvwwzyx8E_8K3opprsoUv0MYITjI"


def test_connection():
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        collections = client.get_collections()
        print("✅ Connected successfully!")
        print("Available collections:", [
              c.name for c in collections.collections])
    except Exception as e:
        print("❌ Connection failed:", e)


if __name__ == "__main__":
    test_connection()
