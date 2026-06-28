from huggingface_hub import HfApi

api = HfApi()

USERNAME = "tumhara-hf-username"
REPO_NAME = "medguard-ai-xray-model"

# Poora folder ek baar mein upload
api.upload_folder(
    folder_path="E:\Taimoor Ahmad (5th 1M)\code",        # local folder
    repo_id=f"{Taimoorahmad1789}/{medguard-Ai}",
    repo_type="model"
)

print("✅ Saare model files upload ho gaye!")