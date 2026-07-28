import requests
from requests.auth import HTTPBasicAuth
import os

class NeuralCloudClient:
    def __init__(self, base_url="http://localhost:8002/remote.php/dav/files/admin", user="admin", password="NeuralAI_Admin_2026!"):
        self.base_url = base_url.rstrip('/')
        self.auth = HTTPBasicAuth(user, password)

    def _get_url(self, path):
        return f"{self.base_url}/{path.lstrip('/')}"

    def list_files(self, path=""):
        url = self._get_url(path)
        headers = {'Depth': '1'}
        response = requests.request('PROPFIND', url, auth=self.auth, headers=headers)
        if response.status_code == 207:
            # Simple XML parsing would be better, but let's keep it minimal for now
            return response.text
        response.raise_for_status()

    def upload_file(self, local_path, remote_path):
        url = self._get_url(remote_path)
        with open(local_path, 'rb') as f:
            response = requests.put(url, auth=self.auth, data=f)
        response.raise_for_status()
        return response.status_code

    def download_file(self, remote_path, local_path):
        url = self._get_url(remote_path)
        response = requests.get(url, auth=self.auth, stream=True)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_path

    def delete_file(self, remote_path):
        url = self._get_url(remote_path)
        response = requests.delete(url, auth=self.auth)
        response.raise_for_status()
        return response.status_code

    def mkdir(self, remote_path):
        url = self._get_url(remote_path)
        response = requests.request('MKCOL', url, auth=self.auth)
        if response.status_code == 405: # Already exists
            return 201
        response.raise_for_status()
        return response.status_code

if __name__ == "__main__":
    client = NeuralCloudClient()
    print("Testing NeuralCloudClient...")
    try:
        # Create a test directory
        client.mkdir("NeuralAI_Vault")
        # Upload a test file
        test_file = "/tmp/cloud_test.txt"
        with open(test_file, "w") as f:
            f.write("NeuralAI Cloud Storage Test")
        client.upload_file(test_file, "NeuralAI_Vault/test.txt")
        print("Upload successful!")
        # List files
        print("Remote files:")
        print(client.list_files("NeuralAI_Vault"))
    except Exception as e:
        print(f"Error: {e}")
